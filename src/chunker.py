import re
import json

SECTION_HEADERS = ["SUMMARY", "EDUCATION", "TECHNICAL SKILLS", "WORK EXPERIENCE", "PROJECTS", "CERTIFICATIONS"]

def split_into_sections(text):
    pattern = "|".join(SECTION_HEADERS)
    matches = list(re.finditer(pattern, text))
    sections = {}
    for i, m in enumerate(matches):
        header = m.group()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[header] = text[start:end].strip()
    return sections

def split_entries(section_text):
    """Splits a WORK EXPERIENCE or PROJECTS section into individual entries,
    accumulating ALL bullets under one title until a new title appears."""
    segments = [s.strip() for s in re.split(r'\s{2,}', section_text) if s.strip()]

    entries = []
    current_entry = None
    pending_title = None

    for i, seg in enumerate(segments):
        is_last = (i == len(segments) - 1)

        if "•" in seg:
            if current_entry is None:
                current_entry = {"title": pending_title, "text": pending_title or ""}
                pending_title = None

            if is_last:
                body, next_title = seg, None
            else:
                last_period = seg.rfind(".")
                if last_period == -1:
                    body, next_title = seg, None
                else:
                    body = seg[:last_period + 1]
                    next_title = seg[last_period + 1:].strip() or None

            current_entry["text"] += " " + body

            if next_title:
                entries.append(current_entry)
                current_entry = None
                pending_title = next_title
        else:
            # Title fragment, no bullets yet
            if current_entry is not None:
                entries.append(current_entry)
                current_entry = None
            pending_title = f"{pending_title} {seg}" if pending_title else seg

    if current_entry is not None:
        entries.append(current_entry)

    return entries

def build_corpus(resume_text):
    sections = split_into_sections(resume_text)
    corpus = []
    chunk_id = 0

    for section_name, category in [("WORK EXPERIENCE", "experience"), ("PROJECTS", "project")]:
        if section_name not in sections:
            continue
        for entry in split_entries(sections[section_name]):
            corpus.append({
                "id": f"chunk_{chunk_id:03d}",
                "category": category,
                "title": entry["title"],
                "text": entry["text"]
            })
            chunk_id += 1

    return corpus

if __name__ == "__main__":
    from parser import extract_text_from_resume
    text = extract_text_from_resume("data/sample_resume.pdf")
    corpus = build_corpus(text)

    for c in corpus:
        print(f"--- {c['id']} | {c['category']} ---")
        print("TITLE:", c["title"])
        print(c["text"][:200], "...\n")

    with open("data/corpus.json", "w") as f:
        json.dump(corpus, f, indent=2)
    print(f"Saved {len(corpus)} chunks to data/corpus.json")