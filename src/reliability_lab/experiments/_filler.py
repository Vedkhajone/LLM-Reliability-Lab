"""Neutral filler paragraphs used for irrelevant (C2) and excessive (C4) context.

They deliberately avoid any of the facts asked about in the dataset. Selection is
deterministic given the case id so that runs are reproducible.
"""
from __future__ import annotations

import zlib

FILLER_PARAGRAPHS: list[str] = [
    "Sourdough bread relies on a fermented starter containing wild yeast and lactic acid bacteria. Bakers often feed the starter on a regular schedule, and the rise time depends strongly on room temperature and the flour used.",
    "Public libraries typically organise their collections using a classification scheme so that books on similar subjects sit near each other. Many libraries also lend digital media, run community workshops and provide quiet study spaces.",
    "Long-distance cyclists commonly plan routes around elevation profiles, water stops and weather forecasts. Carrying a basic repair kit with a spare tube, tyre levers and a small pump is considered standard practice.",
    "A typical municipal recycling programme sorts paper, glass, metal and some plastics into separate streams. Contamination from food residue can reduce the value of the collected material and sometimes leads to whole batches being rejected.",
    "Amateur astronomers often begin with binoculars before moving to a telescope. Dark skies away from city lights matter more for faint objects than a larger aperture used under heavy light pollution.",
    "Spreadsheet software lets users arrange numbers in rows and columns and apply formulas to them. Common features include sorting, filtering, charts and pivot tables for summarising larger lists of records.",
    "Indoor plants vary widely in their light and watering needs. Overwatering is a frequent cause of failure because roots sitting in saturated soil can rot, while many hardy species tolerate occasional neglect.",
    "Train timetables are usually built around a repeating pattern so that services depart at the same minutes past each hour. Planners balance frequency, platform capacity and the time needed for crews to turn around.",
    "Coffee brewing methods differ mainly in grind size, water temperature and contact time. Finer grinds extract faster, so espresso uses a short contact time while cold brew steeps for many hours.",
    "Community gardens give residents a shared space to grow vegetables and flowers. Plots are often allocated by waiting list, and volunteers coordinate composting, tool sharing and seasonal planting days.",
    "Chess openings are sequences of early moves that aim to develop pieces and control the centre. Players memorise common lines, but understanding the underlying plans is generally more useful than rote memorisation.",
    "Home insulation reduces the rate at which heat escapes in winter and enters in summer. Roof spaces and wall cavities are the usual starting points because they account for a large share of thermal loss.",
    "Weather forecasts combine observations from satellites, radar and ground stations with numerical models. Forecast accuracy generally decreases with lead time, so predictions several days ahead carry more uncertainty.",
    "Photography exposure depends on three linked settings: aperture, shutter speed and sensitivity. Changing one setting usually requires adjusting another to keep the image evenly lit.",
    "Local farmers' markets let growers sell directly to customers, often offering seasonal produce that has travelled only a short distance. Stalls may also sell preserves, baked goods and cut flowers.",
    "Basic budgeting methods divide income into categories such as housing, food, transport and savings. Reviewing spending monthly helps people notice patterns and adjust before small overruns accumulate.",
]


def pick_filler(case_id: str, n: int, offset: int = 0) -> list[str]:
    """Deterministically pick `n` filler paragraphs (cycling if n > bank size)."""
    start = (zlib.crc32(case_id.encode("utf-8")) + offset) % len(FILLER_PARAGRAPHS)
    return [FILLER_PARAGRAPHS[(start + i) % len(FILLER_PARAGRAPHS)] for i in range(n)]
