"""
Prompt variants for the system eval's prompt-edit loop.

"baseline" is the production prompt in summarize_video.py, unchanged. Every
other entry overrides the system prompt and/or some section blocks; anything it
leaves out comes from production. A section override must keep its "## "
header line, because the deterministic checks look for the production headers.

To try an edit, add a new entry under a new name. Never change an entry that
has summaries on disk: regen refuses to mix prompts under one --variant name.
"""

import summarize_video as sv

BASE_SYSTEM = sv.SUMMARY_SYSTEM
BASE_SECTIONS = dict(sv.SECTION_PROMPTS)

PROMPTS: dict[str, dict] = {
    "baseline": {},
    # Aimed at the judge's flag categories on the dev split: claims the
    # transcript doesn't support, numbers it never says, and citations to the
    # wrong moment.
    "grounded-v1": {
        "system": (
            "You summarize video transcripts so a reader gets the high points "
            "without watching. The transcript is machine-generated and may have "
            "small errors or missing punctuation; read past those, but do not "
            "repair a garbled word into a specific name, number or product you "
            "cannot see in the text. Every claim must be something the speaker "
            "actually said: no inferred motives, no background knowledge, no "
            "rounding or converting numbers. When a claim maps to a moment in "
            "the video, cite the [HH:MM:SS] marker printed just before the words "
            "you are relying on, never a marker you estimated. If you are unsure "
            "whether something was said, leave it out."
        ),
        "sections": {
            "actionable": BASE_SECTIONS["actionable"]
            + "Copy amounts, doses and settings exactly as spoken. If the speaker\n"
            'gave a range or hedged ("about", "roughly"), keep the hedge.\n',
        },
    },
}


def apply_prompt(name: str) -> None:
    """Point summarize_video at a variant's prompt for the rest of this process.

    summarize_anthropic reads SUMMARY_SYSTEM and SECTION_PROMPTS at call time, so
    swapping the module globals changes what it sends, and prompt_sha() sees the
    same text.
    """
    if name not in PROMPTS:
        raise SystemExit(
            f"unknown --prompt {name!r}; known: {', '.join(PROMPTS)} "
            "(add one in evals/prompts.py)"
        )
    entry = PROMPTS[name]
    sv.SUMMARY_SYSTEM = entry.get("system", BASE_SYSTEM)
    sv.SECTION_PROMPTS = BASE_SECTIONS | entry.get("sections", {})
