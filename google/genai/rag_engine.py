# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for simple RAG workflows."""

import asyncio
import pickle
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import whisper
from pdfminer.high_level import extract_text
from PIL import Image

from . import types, Client


@dataclass
class Evento:
  """Represents an event extracted from a text file."""
  line: int
  text: str


async def extract_zip(zip_path: Path, out_dir: Path) -> Tuple[Path, List[Path]]:
  """Extracts a zip archive asynchronously.

  Args:
    zip_path: Path to the zip archive.
    out_dir: Directory to extract files into.

  Returns:
    A tuple containing the output directory and list of extracted files.
  """

  out_dir.mkdir(parents=True, exist_ok=True)

  def _extract() -> List[str]:
    with zipfile.ZipFile(zip_path, 'r') as zf:
      zf.extractall(out_dir)
      return zf.namelist()

  names = await asyncio.to_thread(_extract)
  files = [out_dir / name for name in names]
  return out_dir, files


async def parse_txt(txt_path: Path) -> List[Evento]:
  """Parses a text file into ``Evento`` objects."""

  text = await asyncio.to_thread(txt_path.read_text, encoding='utf-8')
  return [Evento(i + 1, line) for i, line in enumerate(text.splitlines())]


async def transcribe_audio(opus_path: Path) -> str:
  """Transcribes an opus audio file using Whisper."""

  wav_path = opus_path.with_suffix('.wav')
  cmd = ['ffmpeg', '-y', '-i', str(opus_path), str(wav_path)]
  proc = await asyncio.create_subprocess_exec(
      *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
  )
  await proc.communicate()

  def _transcribe() -> str:
    model = whisper.load_model('base')
    result = model.transcribe(str(wav_path))
    return result['text']

  return await asyncio.to_thread(_transcribe)


async def analyze_media(media_path: Path, client: Client) -> str:
  """Analyzes an image or PDF using Gemini models."""

  suffix = media_path.suffix.lower()
  if suffix in {'.jpg', '.jpeg', '.png', '.webp'}:
    image = await asyncio.to_thread(Image.open, media_path)
    part = types.Part.from_bytes(
        data=await asyncio.to_thread(image.tobytes),
        mime_type=f'image/{suffix.lstrip(".")}'
    )
    contents = types.UserContent([part])
    response = await client.aio.models.generate_content(
        model='gemini-pro-vision', contents=[contents]
    )
    return response.text
  elif suffix == '.pdf':
    text = await asyncio.to_thread(extract_text, str(media_path))
    response = await client.aio.models.generate_content(
        model='gemini-pro', contents=text
    )
    return response.text
  else:
    raise ValueError(f'Unsupported media type: {suffix}')


async def build_index(
    events: List[Evento], media_texts: Dict[Path, str]
) -> Any:
  """Builds and persists an index from events and media texts."""

  index = {
      'events': [event.__dict__ for event in events],
      'media': {str(p): t for p, t in media_texts.items()},
  }
  rag_dir = Path('_whats_rag')
  rag_dir.mkdir(exist_ok=True)
  index_path = rag_dir / 'index.pkl'

  def _dump() -> None:
    with open(index_path, 'wb') as fh:
      pickle.dump(index, fh)

  await asyncio.to_thread(_dump)
  return index

