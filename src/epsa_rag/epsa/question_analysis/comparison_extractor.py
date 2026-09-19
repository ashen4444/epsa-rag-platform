"""Local comparative cues, alternative spans, and conservative semantic targets."""

from __future__ import annotations

import re
from dataclasses import dataclass

from epsa_rag.epsa.question_analysis.config import QuestionAnalyzerConfig
from epsa_rag.epsa.question_analysis.models import EntityMention
from epsa_rag.epsa.question_analysis.normalizer import normalize_entity

_COMPARATIVE = re.compile(
    r"\b(?:more recently|older|younger|earlier|later|first|prior|higher|lower|"
    r"larger|smaller|broader|taller|shorter|greater|more|most|less|fewer|farther|further|"
    r"nearer|closer)\b",
    re.IGNORECASE,
)
_ORDINAL_NOUN = re.compile(r"\s+(?:phase|part|chapter|season|edition|round)\b", re.IGNORECASE)
_QUANTITY_ATTRIBUTE = re.compile(r"\s+(?:[\w#-]+\s+){0,3}[\w#-]+s\b", re.IGNORECASE)
_EVENT_PREFIX = re.compile(
    r"(?:the\s+)?(?:end|start|beginning|release|publication|production|founding|"
    r"death|birth)\s+of\s+$",
    re.IGNORECASE,
)
_PERSON_OF_PLACE = re.compile(
    r"(?P<name>[A-Z][\w'-]+\s+[A-Z][\w'-]+)\s+of\s+[A-Z][\w'-]+$"
)
_NAMED_GROUP_DESCRIPTOR = re.compile(
    r"^[A-Z][\w'-]+(?:\s+[A-Z][\w'-]+)?['\u2019]s\s+(?:band|group)\s+"
)
_QUOTED = re.compile(r'"[^\"]+"|(?<!\w)\'[^\']+\'(?!\w)')
_SEPARATOR = re.compile(r"\b(?:or|and|versus)\b|\bvs\.?(?=\s)", re.IGNORECASE)
_PAIR_INTRO = re.compile(r"\b(?:between|out of|(?:which|who|what) of)\b", re.IGNORECASE)
_AUXILIARY_LEAD = re.compile(r"^\s*(?:is|are|was|were|does|do|did|has|have|had)\b", re.IGNORECASE)
_AND_SELECTOR = re.compile(r"^\s*(?:which|who|what|between|how)\b", re.IGNORECASE)
_NUMBERED_PREFIX = re.compile(r"(?:\d+(?:st|nd|rd|th)?\s+)+$", re.IGNORECASE)
_ROLE = (
    r"(?:director|comedian|singer|actor|actress|astronaut|writer|author|manager|coach|"
    r"composer|engineer|scholar|goalkeeper|politician|journalist)"
)
_ROLE_PREFIX = re.compile(rf"^(?:the\s+)?(?:[A-Z][\w-]*\s+){{0,2}}{_ROLE}\s+", re.IGNORECASE)
_APPOSITIVE = re.compile(rf",\s*(?:the\s+)?(?:[\w-]+\s+){{0,3}}{_ROLE}\s*,?\s*$", re.IGNORECASE)
_LEAD_SCAFFOLD = re.compile(
    r"^(?:out of|between|which of)\s+|^(?:the\s+)?(?:one|two|three)\s+"
    r"(?:[a-z]+\s+)?[,;:]?\s*|^(?:the\s+)?(?:one|two|three)[,;:]\s*|"
    r"^[a-z]+\s*[,;:]\s*"
)
_BAD_ENDING = re.compile(
    r"\b(?:include|has|have|was|were|is|are|lives|released|founded|formed)\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _Cue:
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class _Separator:
    start: int
    end: int
    text: str


def _overlaps(start: int, end: int, spans: tuple[tuple[int, int], ...]) -> bool:
    return any(start < span_end and span_start < end for span_start, span_end in spans)


def _protected_spans(
    question: str, seed_entities: tuple[EntityMention, ...]
) -> tuple[tuple[int, int], ...]:
    quotes = ((match.start(), match.end()) for match in _QUOTED.finditer(question))
    seeds = ((seed.start, seed.end) for seed in seed_entities)
    return (*quotes, *seeds)


def _cues(question: str, protected: tuple[tuple[int, int], ...]) -> tuple[_Cue, ...]:
    result: list[_Cue] = []
    for match in _COMPARATIVE.finditer(question):
        if _overlaps(match.start(), match.end(), protected):
            continue
        if match.group().casefold() == "first" and _ORDINAL_NOUN.match(question, match.end()):
            continue
        if match.group().casefold() == "most" and _QUANTITY_ATTRIBUTE.match(
            question, match.end()
        ) is None:
            continue
        if match.group().casefold() == "first" and question[: match.start()].casefold().endswith(
            "the "
        ):
            clause_end = re.search(r"[,;:?]", question[match.end() :])
            tail = (
                question[match.end() : match.end() + clause_end.start()]
                if clause_end
                else question[match.end() :]
            )
            if re.search(r"\bto\b", tail, re.IGNORECASE):
                continue
        result.append(_Cue(match.start(), match.end(), match.group()))
    return tuple(result)


def _separators(question: str, protected: tuple[tuple[int, int], ...]) -> tuple[_Separator, ...]:
    return tuple(
        _Separator(match.start(), match.end(), match.group().casefold())
        for match in _SEPARATOR.finditer(question)
        if not _overlaps(match.start(), match.end(), protected)
    )


def _nearby(
    cue: _Cue,
    separator: _Separator,
    question: str,
    protected: tuple[tuple[int, int], ...],
) -> bool:
    if cue.end <= separator.start:
        lower, upper = cue.end, separator.start
    elif separator.end <= cue.start:
        lower, upper = separator.end, cue.start
    else:
        return False
    return upper - lower <= 120 and not any(
        question[index] in "?;!"
        and not _overlaps(index, index + 1, protected)
        and not (question[index] == "!" and _overlaps(index - 1, index, protected))
        for index in range(lower, upper)
    )


def _first_seed(
    seeds: tuple[EntityMention, ...], lower: int, upper: int
) -> EntityMention | None:
    candidates = [seed for seed in seeds if lower <= seed.start and seed.end <= upper]
    return min(candidates, key=lambda seed: seed.start) if candidates else None


def _last_seed(seeds: tuple[EntityMention, ...], lower: int, upper: int) -> EntityMention | None:
    candidates = [seed for seed in seeds if lower <= seed.start and seed.end <= upper]
    return max(candidates, key=lambda seed: seed.end) if candidates else None


def _include_prefix(question: str, lower: int, seed_start: int) -> int:
    prefix = question[lower:seed_start]
    event = _EVENT_PREFIX.search(prefix)
    if event is not None:
        return lower + event.start()
    numbered = _NUMBERED_PREFIX.search(prefix)
    if numbered is not None:
        return lower + numbered.start()
    article = re.search(r"\bthe\s+$", prefix, re.IGNORECASE)
    return lower + article.start() if article is not None else seed_start


def _left_start(
    question: str,
    cue: _Cue,
    separator: _Separator,
    seeds: tuple[EntityMention, ...],
) -> int:
    cue_before = cue.end <= separator.start
    lower = cue.end if cue_before else 0
    prefix = question[lower : separator.start]
    introductions = tuple(_PAIR_INTRO.finditer(prefix))
    if introductions:
        intro = introductions[-1]
        # A leading "which of" may describe the answer slot, followed by a later comma pair.
        if not cue_before or intro.end() + lower > cue.end:
            if not (cue_before and intro.group().casefold() in {"which of", "who of", "what of"}):
                return lower + intro.end()
    if cue_before:
        seed = _first_seed(seeds, lower, separator.start)
        punctuation = re.search(r"[:,]", prefix)
        if punctuation is not None:
            absolute = lower + punctuation.start()
            attribute_tail = (
                seed is not None
                and (
                    (seed.end <= absolute and bool(question[seed.end:absolute].strip()))
                    or (
                        cue.text.casefold() in {"closer", "nearer", "farther", "further"}
                        and re.search(r"\bto\b", question[lower:seed.start]) is not None
                    )
                )
            )
            if seed is None or absolute < seed.start or attribute_tail:
                return lower + punctuation.end()
        return _include_prefix(question, lower, seed.start) if seed is not None else lower
    auxiliary = _AUXILIARY_LEAD.match(question)
    if auxiliary is not None:
        return auxiliary.end()
    seed = _first_seed(seeds, 0, separator.start)
    return _include_prefix(question, 0, seed.start) if seed is not None else 0


def _right_end(
    question: str,
    cue: _Cue,
    separator: _Separator,
    seeds: tuple[EntityMention, ...],
) -> int:
    upper = cue.start if separator.end <= cue.start else len(question)
    seed = _last_seed(seeds, separator.end, upper)
    if seed is not None:
        suffix = question[seed.end:upper].strip(" ?")
        if suffix in {"'s", "\u2019s"}:
            return seed.end + 2
        if re.fullmatch(r"(?:\s+[a-z][\w-]*){1,2}\s*[?]?\s*", question[seed.end:upper]):
            if not re.search(r"\b(?:lives|include|has|have|was|were|is|are|formed)\b", suffix):
                return upper
        return seed.end
    return upper


def _strip_edges(question: str, start: int, end: int) -> tuple[int, int]:
    edge = " \t\r\n,;:?\"'\u2013\u2014"
    while start < end and question[start] in edge:
        start += 1
    while start < end and question[end - 1] in edge:
        end -= 1
    return start, end


def _clean_span(
    question: str,
    start: int,
    end: int,
    seeds: tuple[EntityMention, ...],
    *,
    left: bool,
) -> tuple[int, int]:
    start, end = _strip_edges(question, start, end)
    if left:
        appositive = _APPOSITIVE.search(question[start:end])
        if appositive is not None:
            end = start + appositive.start()
    while True:
        scaffold = _LEAD_SCAFFOLD.match(question[start:end])
        if scaffold is None or scaffold.end() == 0:
            break
        start += scaffold.end()
        start, end = _strip_edges(question, start, end)
    role_prefix = _ROLE_PREFIX.match(question[start:end])
    if role_prefix is not None:
        later_seed = _first_seed(seeds, start + role_prefix.end(), end)
        if later_seed is not None:
            start += role_prefix.end()
    group_descriptor = _NAMED_GROUP_DESCRIPTOR.match(question[start:end])
    if group_descriptor is not None and _first_seed(seeds, start + group_descriptor.end(), end):
        start += group_descriptor.end()
    candidates = [seed for seed in seeds if start <= seed.start and seed.end <= end]
    if candidates:
        first = min(candidates, key=lambda seed: seed.start)
        strong_later = [
            seed
            for seed in candidates
            if seed.start > first.start and seed.confidence > first.confidence
        ]
        if strong_later and (
            re.search(r"\b(?:band|group)\b", question[first.end : strong_later[0].start], re.I)
            or (
                first.source == "title_like_token"
                and re.fullmatch(
                    r"\s+(?:in|from|for)\s+(?:[a-z]+\s+){1,3}",
                    question[first.end : strong_later[0].start],
                    re.I,
                )
            )
        ):
            start = strong_later[0].start
        else:
            suffix = question[first.end:end]
            if first.start == start and re.fullmatch(r"\s+of\s+[A-Z][\w-]+", suffix):
                end = first.end
    person_of_place = _PERSON_OF_PLACE.fullmatch(question[start:end])
    if person_of_place is not None:
        end = start + person_of_place.end("name")
    return _strip_edges(question, start, end)


def _target(
    question: str,
    start: int,
    end: int,
    seeds: tuple[EntityMention, ...],
    *,
    source: str,
    score: float,
    left: bool,
) -> EntityMention | None:
    start, end = _clean_span(question, start, end, seeds, left=left)
    text = question[start:end]
    if not text or len(text) > 150:
        return None
    if _BAD_ENDING.search(text) and not any(
        seed.start == start and seed.end == end for seed in seeds
    ):
        return None
    supported = any(
        (start <= seed.start and seed.end <= end)
        or (
            seed.start < start < seed.end <= end
            and (
                question[seed.start:start].casefold().strip() in {"of", "the"}
                or question[seed.start:start].rstrip().endswith(",")
            )
        )
        or (
            start <= seed.start < end < seed.end
            and (
                question[end:seed.end].lstrip().startswith(",")
                or (
                    seed.start == start
                    and question[end:seed.end].startswith(" of ")
                    and _PERSON_OF_PLACE.fullmatch(question[seed.start:seed.end]) is not None
                )
            )
        )
        for seed in seeds
    )
    if not supported:
        return None
    return EntityMention(
        text=text,
        normalized=normalize_entity(text),
        source=source,
        start=start,
        end=end,
        confidence=score,
    )


def analyze_comparison(
    matching_question: str,
    seed_entities: tuple[EntityMention, ...],
    config: QuestionAnalyzerConfig,
) -> tuple[EntityMention, EntityMention] | None:
    """Find a comparative cue governing two local alternatives and return clean targets."""

    protected = _protected_spans(matching_question, seed_entities)
    cues = _cues(matching_question, protected)
    separators = _separators(matching_question, protected)
    for cue in cues:
        ordered = sorted(
            separators,
            key=lambda separator: (
                separator.text == "and",
                abs(separator.start - cue.start),
            ),
        )
        for separator in ordered:
            if not _nearby(cue, separator, matching_question, protected):
                continue
            if separator.text == "and" and _AND_SELECTOR.match(matching_question) is None:
                continue
            if separator.text == "and" and re.match(
                r"\s*(?:now|then|whose|which|that|who)\b",
                matching_question[separator.end :],
                re.IGNORECASE,
            ):
                continue
            left_start = _left_start(matching_question, cue, separator, seed_entities)
            right_end = _right_end(matching_question, cue, separator, seed_entities)
            source = (
                "comparison_between"
                if "between" in matching_question[: separator.start].casefold()
                else "comparison_which_of"
            )
            score = (
                config.between_comparison_score
                if source == "comparison_between"
                else config.which_of_comparison_score
            )
            left = _target(
                matching_question,
                left_start,
                separator.start,
                seed_entities,
                source=source,
                score=score,
                left=True,
            )
            right = _target(
                matching_question,
                separator.end,
                right_end,
                seed_entities,
                source=source,
                score=score,
                left=False,
            )
            if left is not None and right is not None and left.normalized != right.normalized:
                return left, right
    return None
