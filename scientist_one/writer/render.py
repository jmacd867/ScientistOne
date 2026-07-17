from typing import Protocol


class Renderer(Protocol):
    def render(self, title: str, body_md: str, references: list[dict]) -> str: ...


class MarkdownRenderer:
    def render(self, title: str, body_md: str, references: list[dict]) -> str:
        parts = [f"# {title}", "", body_md.strip()]
        if references:
            parts += ["", "## References", ""]
            for i, ref in enumerate(references, 1):
                authors = ", ".join(ref.get("authors") or []) or "Unknown"
                parts.append(f"{i}. {authors} ({ref.get('year')}). "
                             f"{ref['title']}. {ref.get('url', '')}")
        return "\n".join(parts) + "\n"
