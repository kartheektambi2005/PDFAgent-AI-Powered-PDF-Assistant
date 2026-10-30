Project Instructions

General Rules

- Keep the code simple, clean, and maintainable.
- Avoid unnecessary files, dependencies, and complex architecture.
- Follow a modular structure.
- Explain important implementation decisions briefly.

Token Optimization

- Minimize unnecessary Gemini API calls.
- Never send the entire PDF for every question.
- Reuse stored embeddings.
- Retrieve only relevant chunks.
- Keep prompts and conversation history concise.
- Track token usage whenever supported.

Hallucination Prevention

- Generate answers using retrieved PDF context only.
- Never invent missing information.
- If sufficient evidence is unavailable, clearly say so.
- Include source page references whenever possible.

Development Rules

- Inspect existing code before modifying it.
- Do not overwrite working code unnecessarily.
- Handle errors properly.
- Test important functionality before claiming completion.
- Never expose API keys or secrets in source code.