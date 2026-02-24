"""
skills/loader.py — Skill Discovery and Loader

Scans skill directories, imports Python modules, finds SkillBase subclasses,
validates their manifests, instantiates them, and populates a SkillRegistry.

Rules:
  - A skill is any module-level class that subclasses SkillBase and has a manifest.
  - SkillBase itself is excluded (abstract).
  - Duplicate skill names across files raise SkillValidationError at startup.
  - A module that fails to import is logged as a warning — other skills still load.
  - Call load_all() once at startup. The returned SkillRegistry is read-only at runtime.

Usage:
    loader = SkillLoader()
    registry = loader.load_all([
        Path("skills/builtin"),
        Path("skills/plugins"),
    ])
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Optional

from skills.base import SkillBase
from skills.registry import SkillRegistry
from skills.types import SkillValidationError

try:
    from observability.logger import get_logger as _get_logger
    _log_raw = _get_logger(__name__)
    _STRUCTLOG = True
except Exception:
    import logging as _logging
    _log_raw = _logging.getLogger(__name__)
    _STRUCTLOG = False


def _log(level, event, **kwargs):
    if _STRUCTLOG:
        getattr(_log_raw, level)(event, **kwargs)
    else:
        extra = " ".join(f"{k}={v}" for k, v in kwargs.items())
        getattr(_log_raw, level)("%s %s", event, extra)


class SkillLoader:
    """
    Discovers and loads skills from one or more directories.

    Designed to be called once at startup. Not thread-safe for concurrent
    load_all() calls.
    """

    def load_all(
        self,
        skill_dirs: list[Path],
        registry: Optional[SkillRegistry] = None,
        strict: bool = False,
    ) -> SkillRegistry:
        """
        Discover, validate, and register all skills found in skill_dirs.

        Args:
            skill_dirs: Directories to scan. Scanned in order; earlier dirs
                        take priority if two files define a skill with the same
                        name (raises SkillValidationError — names must be unique).
            registry:   Existing registry to populate. If None, a new one is created.
            strict:     If True, a module-level import error in a skill file raises
                        SkillValidationError immediately rather than logging a warning
                        and continuing. Use strict=True in production deployments so
                        a misconfigured environment fails loudly at startup instead of
                        silently running with missing skills.

        Returns:
            Populated SkillRegistry.

        Raises:
            SkillValidationError: if a skill has a bad manifest, a duplicate name,
                                  or (when strict=True) a module import error.
        """
        reg = registry or SkillRegistry()
        loaded_count = 0
        skipped_count = 0

        for skill_dir in skill_dirs:
            skill_dir = Path(skill_dir)
            if not skill_dir.exists():
                _log("debug", "skill_loader.dir_not_found", path=str(skill_dir))
                continue

            for py_file in sorted(skill_dir.glob("*.py")):
                if py_file.name.startswith("_"):
                    continue  # skip __init__.py, __pycache__, etc.

                skill_classes = self._load_module(py_file, strict=strict)
                for cls in skill_classes:
                    try:
                        cls._validate_manifest()
                        instance = cls()
                        reg.register(instance)
                        _log("debug",
                            "skill_loader.registered",
                            skill=cls.manifest.name,
                            version=cls.manifest.version,
                            file=py_file.name,
                        )
                        loaded_count += 1
                    except (SkillValidationError, ValueError) as e:
                        # Bad manifest or duplicate name — fail loudly at startup
                        raise SkillValidationError(
                            f"Failed to load skill from {py_file}: {e}"
                        ) from e

        _log("info",
            "skill_loader.complete",
            loaded=loaded_count,
            skipped=skipped_count,
            dirs=[str(d) for d in skill_dirs],
        )
        return reg

    def _load_module(self, py_file: Path, strict: bool = False) -> list[type[SkillBase]]:
        """
        Import a Python file and return all SkillBase subclasses defined in it.

        Args:
            py_file: Path to the skill module file.
            strict:  If True, re-raise import errors as SkillValidationError
                     instead of logging and skipping.

        Returns:
            List of SkillBase subclasses found in the module. Empty list if the
            file fails to import and strict=False.
        """
        module_name = f"_skill_{py_file.stem}_{id(py_file)}"
        spec = importlib.util.spec_from_file_location(module_name, py_file)
        if spec is None or spec.loader is None:
            _log("warning", "skill_loader.bad_spec", file=str(py_file))
            return []

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module

        try:
            spec.loader.exec_module(module)  # type: ignore[union-attr]
        except Exception as e:
            if strict:
                # Fail loudly — re-raise so the agent refuses to start with a
                # broken skill environment rather than silently losing tools.
                raise SkillValidationError(
                    f"Failed to import skill module {py_file}: {e}"
                ) from e
            # Import error in a skill file — warn and skip, don't crash startup
            try:
                _log("warning",
                    "skill_loader.import_error",
                    file=str(py_file),
                    error=str(e),
                )
            except TypeError:
                # Fallback: stdlib logger doesn't accept keyword kwargs
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "skill_loader.import_error: file=%s error=%s", py_file, e
                )
            sys.modules.pop(module_name, None)
            return []

        # Find all SkillBase subclasses defined in this module
        classes = []
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, SkillBase)
                and obj is not SkillBase
                and obj.__module__ == module_name
                and hasattr(obj, "manifest")
            ):
                classes.append(obj)

        return classes

    def discover_classes(self, skill_dirs: list[Path]) -> list[type[SkillBase]]:
        """
        Return all discovered SkillBase subclasses without registering them.
        Useful for introspection and testing.
        """
        all_classes = []
        for skill_dir in skill_dirs:
            for py_file in sorted(Path(skill_dir).glob("*.py")):
                if not py_file.name.startswith("_"):
                    all_classes.extend(self._load_module(py_file))
        return all_classes