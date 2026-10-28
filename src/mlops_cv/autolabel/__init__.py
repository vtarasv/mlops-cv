"""The auto-labeler: zero-shot detector -> box filter -> segmenter, scored against VisDrone GT.

Experiment code only: nothing in production imports it. Stage configs live under
``configs/autolabel/``; ``docs/autolabel.md`` records each stage's command and results.
"""
