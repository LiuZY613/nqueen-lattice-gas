# Thesis Structure Refactor Design

**Date:** 2026-04-08

**Status:** Approved in chat, pending implementation

## Goal

Refactor the current single-file LaTeX sample into a long-form thesis writing structure centered on a formal entry file named `thesis.tex`, with each body chapter stored in its own `.tex` file and appendices split into separate files as well.

The design should support a thesis on the order of 100 pages without forcing the author to work in one oversized source file.

## Current State

- The repository currently uses a single entry file: `njuthesis-sample.tex`.
- The sample file contains:
  - document class and global setup
  - front matter content
  - body content
  - bibliography entry point
  - acknowledgement
  - appendix content
- The template itself already suggests splitting chapters into separate files and including them from the main document.

## Requirements

- Rename the main compilation entry from `njuthesis-sample.tex` to a more formal `thesis.tex`.
- Keep `njuthesis-setup.def` as the central configuration file.
- Split each body chapter into its own `.tex` file.
- Split appendices into separate `.tex` files.
- Keep the compilation workflow centered on a single main file.
- Preserve compatibility with the existing `njuthesis` template structure.
- Avoid unnecessary changes to bibliography and class files.

## Non-Goals

- No changes to `njuthesis.cls` or other class definition files.
- No bibliography file rename in this refactor.
- No over-fragmentation to section-level files.
- No redesign of the thesis template itself.

## Options Considered

### Option 1: Minimal split

Keep most content in the main file and only move body chapters plus appendices into separate files.

Pros:
- Minimal changes
- Very low migration risk

Cons:
- Front matter remains crowded in the main file
- Less clean as the thesis grows

### Option 2: Balanced split

Use a formal `thesis.tex` main file and split content into `frontmatter/`, `chapters/`, and `appendices/`.

Pros:
- Clear structure for long documents
- Keeps the main file focused on orchestration
- Matches how large LaTeX theses are commonly maintained

Cons:
- Slightly more files than the minimal approach

### Option 3: Fine-grained split

Split not only chapters, but also sections and other small units into many files.

Pros:
- Maximum modularity

Cons:
- Too fragmented for everyday writing
- Higher navigation overhead

## Chosen Approach

Choose Option 2: balanced split.

This keeps the repository simple while still making long-form writing manageable. It follows the template's intended usage without introducing extra abstraction.

## Target File Layout

```text
thesis.tex
njuthesis-setup.def
njuthesis-sample.bib
frontmatter/
  abstract-zh.tex
  abstract-en.tex
  acknowledgement.tex
chapters/
  01-introduction.tex
  02-related-work.tex
  03-method.tex
  04-experiments.tex
  05-conclusion.tex
appendices/
  appendix-a.tex
```

## Main File Responsibilities

`thesis.tex` should only contain:

- document class declaration and template options
- `\input{njuthesis-setup.def}`
- optional package imports and custom macros
- front matter assembly
- table of contents and other lists
- body chapter ordering
- bibliography entry point
- acknowledgement inclusion
- appendix switching and appendix inclusion

It should not contain large blocks of chapter prose.

## Inclusion Strategy

### Front matter

Use `\input` for front matter files:

- `frontmatter/abstract-zh.tex`
- `frontmatter/abstract-en.tex`
- `frontmatter/acknowledgement.tex`

Reason:
- these files are short
- they do not need chapter-level page management
- keeping them separate improves readability without adding structural overhead

### Body chapters

Use `\include` for body chapters:

- `chapters/01-introduction`
- `chapters/02-related-work`
- `chapters/03-method`
- `chapters/04-experiments`
- `chapters/05-conclusion`

Reason:
- suitable for large chapter-sized units
- supports `\includeonly{...}` for faster partial compilation
- makes chapter reordering and independent editing straightforward

Each chapter file should contain its own `\chapter{...}` declaration.

### Appendices

Use `\appendix` in `thesis.tex`, then `\include` appendix files such as:

- `appendices/appendix-a`

Each appendix file should contain its own `\chapter{...}` declaration.

## Naming Conventions

- Main file: `thesis.tex`
- Chapter files: numeric prefix plus slug, for example `01-introduction.tex`
- Appendix files: semantic appendix name, for example `appendix-a.tex`

Rationale:
- stable lexical ordering
- easier chapter reordering
- obvious mapping between source files and thesis structure

## Migration Plan

1. Create `frontmatter/`, `chapters/`, and `appendices/`.
2. Rename the current entry file from `njuthesis-sample.tex` to `thesis.tex`.
3. Extract Chinese abstract into `frontmatter/abstract-zh.tex`.
4. Extract English abstract into `frontmatter/abstract-en.tex`.
5. Extract acknowledgement into `frontmatter/acknowledgement.tex`.
6. Move the current sample chapter content into `chapters/01-introduction.tex`.
7. Create placeholder chapter files for the remaining planned chapters.
8. Move the current appendix sample content into `appendices/appendix-a.tex`.
9. Replace inline content in `thesis.tex` with `\input` and `\include` statements.
10. Compile `thesis.tex` to confirm the refactor preserved buildability.

## Author Workflow After Refactor

- Edit `thesis.tex` only when changing global document structure.
- Write daily content in chapter files under `chapters/`.
- Add appendices under `appendices/`.
- Continue compiling from `thesis.tex`.
- Use `\includeonly{...}` later if compile times become inconvenient.

## Risks And Mitigations

### Risk: broken compilation after split

Mitigation:
- keep changes structural only
- move content with minimal rewriting
- verify compilation from `thesis.tex`

### Risk: user writes content back into the main file

Mitigation:
- leave a brief comment in `thesis.tex` explaining its role as the orchestration file

### Risk: file naming drift over time

Mitigation:
- establish numbering convention from the beginning

## Acceptance Criteria

- A formal main file `thesis.tex` exists and is the sole compilation entry point.
- Body chapters are split into separate files under `chapters/`.
- Appendices are split into separate files under `appendices/`.
- Front matter is separated into focused files under `frontmatter/`.
- The repository still builds through the `njuthesis` template workflow.
- The structure is easy to extend to a 100-page thesis without returning to a monolithic source file.
