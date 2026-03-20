# PDF → Obsidian Wiki Generator (v2)

## Overview

This project converts academic PDFs (slides, textbooks, notes) into structured Obsidian markdown notes.

The system extracts:
- concepts
- facts
- relationships (basic)

Then generates:
- wiki-style pages
- optional cheatsheets / flashcards

---

## Design Philosophy

This is a pipeline-based system with strict separation of concerns.

Key principles:

- Each stage does ONE thing
- LLMs are only used where necessary
- Prefer simple, deterministic logic over "smart" systems
- Avoid overengineering

---

## Pipeline

PDFs → Chunks → Facts → Concepts → Pages → Markdown

---

## Current Scope (v2)

We intentionally limit scope:

- Input: English academic PDFs
- Output: "good enough" study notes
- No perfect deduplication
- No complex knowledge graphs

---

## Non-Goals

- Perfect semantic understanding
- Full Wikipedia-style linking
- Handling all document types
- Maximum performance optimization

---

## Development Approach

- Build in small vertical slices
- Implement one function at a time
- Keep files small and readable
- Avoid large abstractions

---

## Project Structure

src/
ingest/ # PDF loading + chunking
extract/ # LLM-based fact extraction
transform/ # grouping + normalization
generate/ # page generation
output/ # writing markdown files


---

## Status

v2 rewrite in progress