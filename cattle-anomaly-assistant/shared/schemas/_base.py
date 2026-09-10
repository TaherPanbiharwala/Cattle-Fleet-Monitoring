"""Every model in this package inherits from here.

Pydantic's default (`extra="ignore"`) silently drops unrecognized fields
instead of raising — so a typo'd or renamed field on an optional value (e.g.
`hered_isolation_score` instead of `herd_isolation_score`) would construct
successfully with the field silently defaulting to `None`, no error anywhere.
PRD Section 6 calls this "the shared contract — freeze this before writing
pipeline code"; a frozen contract should fail loudly on anything outside it.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
