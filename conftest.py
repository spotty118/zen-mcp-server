"""Pytest configuration for the Zen MCP Server project.

This repository's test-suite makes extensive use of ``async def`` tests via
``@pytest.mark.asyncio``.  Normally ``pytest-asyncio`` provides the event loop
integration required for those tests, but the execution environment used for
these challenges doesn't install optional development dependencies.  Without a
plugin Pytest falls back to its synchronous runner and immediately aborts the
first ``async`` test with the familiar error message:

```
Failed: async def functions are not natively supported.
You need to install a suitable plugin for your async framework
```

To keep the project self-contained we provide a minimal fallback plugin that
detects when ``pytest-asyncio`` is unavailable and, in that case, runs
coroutine tests inside a fresh ``asyncio`` event loop.  The implementation is
small but mirrors the behaviour that the real plugin would expose which keeps
the tests – and therefore the production code paths they exercise – identical
to their intended execution.

If ``pytest-asyncio`` *is* installed the hook simply defers to it.
"""

from __future__ import annotations

import asyncio
import importlib.util
import inspect

import pytest


_HAS_PYTEST_ASYNCIO = importlib.util.find_spec("pytest_asyncio") is not None


if not _HAS_PYTEST_ASYNCIO:
    def pytest_addoption(parser: pytest.Parser) -> None:
        """Provide the ``asyncio_mode`` ini option expected by pytest-asyncio."""

        parser.addini(
            "asyncio_mode",
            "Compatibility shim for environments without pytest-asyncio installed",
            default="auto",
        )

    @pytest.hookimpl(tryfirst=True)
    def pytest_pyfunc_call(pyfuncitem: pytest.Function) -> bool | None:
        """Run ``async def`` tests without requiring pytest-asyncio.

        Pytest calls this hook for every collected test function.  When the
        function object is a coroutine definition we create a dedicated event
        loop, execute the coroutine, and perform the necessary clean-up.  A
        truthy return value tells Pytest that the test has been executed and no
        further processing is required.
        """

        test_function = pyfuncitem.obj
        if not inspect.iscoroutinefunction(test_function):
            return None

        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            funcargs = pyfuncitem.funcargs
            testargs = {arg: funcargs[arg] for arg in pyfuncitem._fixtureinfo.argnames}
            loop.run_until_complete(test_function(**testargs))
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:  # pragma: no cover - defensive clean-up
                pass
            asyncio.set_event_loop(None)
            loop.close()

        return True
