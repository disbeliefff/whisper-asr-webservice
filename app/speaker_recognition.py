import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from huggingface_hub import login


@dataclass
class SpeakerProfile:
    name: str
    embedding: List[float]


class SpeakerRecognizer:
    def __init__(
        self,
        model_id: str,
        use_auth_token: Optional[str],
        device: str,
        store_path: str,
        match_threshold: float = 0.75,
    ):
        self.device = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
        self.model_id = model_id
        self.match_threshold = match_threshold
        self.store_path = store_path

        if use_auth_token:
            try:
                login(token=use_auth_token)
            except Exception:
                pass

        self._load_model()
        self.speakers: Dict[str, SpeakerProfile] = {}
        self._load_store()

    def _load_model(self):
        from transformers import AutoModelForAudioXVector, AutoFeatureExtractor
        self.feat_extractor = AutoFeatureExtractor.from_pretrained(self.model_id)
        self.embed_model = AutoModelForAudioXVector.from_pretrained(self.model_id).to(self.device)
        self.embed_model.eval()

    def _load_store(self):
        if os.path.exists(self.store_path):
            try:
                with open(self.store_path, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                    for name, data in obj.items():
                        self.speakers[name] = SpeakerProfile(name=name, embedding=data["embedding"]) 
            except Exception:
                self.speakers = {}

    def _save_store(self):
        os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
        obj = {name: {"embedding": prof.embedding} for name, prof in self.speakers.items()}
        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump(obj, f)

    @torch.inference_mode()
    def _audio_to_embedding(self, audio: np.ndarray, sr: int) -> np.ndarray:
        waveform = torch.tensor(audio, dtype=torch.float32, device=self.device)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        # models expect 16k typically; resample if needed
        if sr != 16000:
            try:
                import torchaudio
                waveform = torchaudio.functional.resample(waveform, sr, 16000)
                _sr = 16000
            except Exception as e:
                raise RuntimeError("Speaker recognition requires 16k audio; either set SAMPLE_RATE=16000 or install torchaudio") from e
        else:
            _sr = sr
        inputs = self.feat_extractor(waveform.squeeze(0).cpu().numpy(), sampling_rate=_sr, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        emb = self.embed_model(**inputs).embeddings
        emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb.squeeze(0).detach().cpu().numpy()

    def enroll(self, name: str, audio: np.ndarray, sr: int) -> SpeakerProfile:
        emb = self._audio_to_embedding(audio, sr)
        self.speakers[name] = SpeakerProfile(name=name, embedding=emb.tolist())
        self._save_store()
        return self.speakers[name]

    def delete(self, name: str) -> bool:
        if name in self.speakers:
            del self.speakers[name]
            self._save_store()
            return True
        return False

    def list(self) -> List[str]:
        return list(self.speakers.keys())

    def match_embedding(self, emb: np.ndarray) -> Tuple[Optional[str], float]:
        if not self.speakers:
            return None, 0.0
        # cosine similarity
        names = []
        scores = []
        for name, prof in self.speakers.items():
            ref = np.asarray(prof.embedding, dtype=np.float32)
            score = float(np.dot(emb, ref) / (np.linalg.norm(emb) * np.linalg.norm(ref) + 1e-8))
            names.append(name)
            scores.append(score)
        idx = int(np.argmax(scores))
        best_name = names[idx]
        best_score = scores[idx]
        if best_score >= self.match_threshold:
            return best_name, best_score
        return None, best_score

    def _infer_segment_speaker_label(self, seg) -> Optional[str]:
        # Try segment-level first
        if isinstance(seg, dict):
            spk = seg.get("speaker")
        else:
            spk = getattr(seg, "speaker", None)
        if spk is not None:
            return spk
        # Fallback to majority speaker from words
        words = seg.get("words") if isinstance(seg, dict) else getattr(seg, "words", None)
        if not words:
            return None
        counts: Dict[str, int] = {}
        for w in words:
            spw = w.get("speaker") if isinstance(w, dict) else getattr(w, "speaker", None)
            if spw is None:
                continue
            counts[spw] = counts.get(spw, 0) + 1
        if not counts:
            return None
        return sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[0][0]

    def annotate_segment_speakers(self, result: dict, audio: np.ndarray, sr: int) -> dict:
        # Expect result to contain diarization speaker labels per word or segment
        # We will pool embeddings per diarization speaker by slicing audio per segment
        if "segments" not in result:
            return result
        segments = result["segments"]
        # Build per-speaker time ranges
        speaker_ranges: Dict[str, List[Tuple[float, float]]] = {}
        for seg in segments:
            spk = self._infer_segment_speaker_label(seg)
            start = float(getattr(seg, "start", 0.0) if not isinstance(seg, dict) else seg.get("start", 0.0))
            end = float(getattr(seg, "end", 0.0) if not isinstance(seg, dict) else seg.get("end", 0.0))
            if spk is None:
                continue
            speaker_ranges.setdefault(spk, []).append((start, end))

        # Compute embeddings per diarization speaker by concatenating their ranges
        spk_to_identity: Dict[str, Tuple[Optional[str], float]] = {}
        for spk, ranges in speaker_ranges.items():
            clips = []
            for (s, e) in ranges:
                b = max(0, int(s * sr))
                e_idx = min(len(audio), int(e * sr))
                if e_idx > b:
                    clips.append(audio[b:e_idx])
            if not clips:
                continue
            concat = np.concatenate(clips)
            emb = self._audio_to_embedding(concat, sr)
            name, score = self.match_embedding(emb)
            spk_to_identity[spk] = (name, score)

        # annotate segments with recognized name and score
        for seg in segments:
            spk = self._infer_segment_speaker_label(seg)
            if spk is None:
                continue
            name_score = spk_to_identity.get(spk)
            if not name_score:
                continue
            name, score = name_score
            if isinstance(seg, dict):
                seg["speaker_recognized"] = name
                seg["speaker_score"] = float(score)
                if name is not None:
                    seg["speaker"] = name
            else:
                setattr(seg, "speaker_recognized", name)
                setattr(seg, "speaker_score", float(score))
                if name is not None:
                    setattr(seg, "speaker", name)
        result["segments"] = segments
        return result 