"""CLI for downloading FRBench release assets."""
from __future__ import annotations

import argparse
import sys

import frbench
from frbench import _config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download FRBench release assets into the local cache.",
    )
    parser.add_argument("assets", nargs="*", help="Manifest asset keys to download")
    parser.add_argument("--list", action="store_true", help="List all asset keys and exit")
    parser.add_argument("--list-models", action="store_true", help="List FR models (exclude detectors)")
    parser.add_argument("--all", action="store_true", help="Download every asset in the manifest")
    parser.add_argument("--force", action="store_true", help="Re-download even if cached")
    parser.add_argument("--refresh", action="store_true", help="Re-download manifest.json before other actions")
    parser.add_argument("--quiet", action="store_true", help="Disable progress bars and logs")
    parser.add_argument(
        "--set",
        metavar="KEY=VALUE",
        action="append",
        default=[],
        dest="set_options",
        help="Persist a setting to the config file, e.g. --set cache=/data/frbench "
        "or --set download_verbose=false (repeatable)",
    )
    parser.add_argument(
        "--unset",
        metavar="KEY",
        action="append",
        default=[],
        dest="unset_options",
        help="Remove a persisted setting from the config file (repeatable)",
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Show resolved settings, their sources, and the config file path",
    )
    args = parser.parse_args(argv)

    if args.quiet:
        frbench.set_verbose(False)

    config_only = (args.set_options or args.unset_options or args.show_config) and not (
        args.assets or args.all or args.list or args.list_models or args.refresh
    )

    if args.set_options or args.unset_options:
        try:
            updates = {}
            for item in args.set_options:
                key, sep, value = item.partition("=")
                if not sep or not key:
                    print(f"Error: --set expects KEY=VALUE, got '{item}'", file=sys.stderr)
                    return 1
                updates[key.strip()] = value
            for key in args.unset_options:
                updates[key.strip()] = None
            frbench.configure(**updates, persist=True)
        except (frbench.FRBenchConfigError, TypeError) as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        print(f"Updated {_config.config_file_path()}")

    if args.show_config or args.set_options or args.unset_options:
        print(f"Config file: {_config.config_file_path()}")
        for key, entry in _config.describe_settings().items():
            print(f"  {key:<17} = {entry['value']!r:<30} ({entry['source']})")

    if config_only:
        return 0

    from .utils.update_check import check_for_updates

    check_for_updates()

    if args.refresh:
        frbench.refresh_manifest()

    if args.list:
        names = frbench.list_assets()
        if not names:
            print("Could not load manifest. Check FRBENCH_REPO / FRBENCH_RELEASE.", file=sys.stderr)
            return 1
        print(f"Cache:   {frbench.CACHE}")
        print(f"Release: {frbench.REPO} @ {frbench.RELEASE}")
        print(f"Assets ({len(names)}):")
        for name in names:
            print(f"  {name}")
        return 0

    if args.list_models:
        models = frbench.list_models()
        if not models:
            print("Could not load manifest.", file=sys.stderr)
            return 1
        print(f"Models ({len(models)}):")
        for m in models:
            print(f"  {m.key}  ({m.backbone} / {m.loss} / {m.dataset})")
        return 0

    if not args.all and not args.assets:
        parser.print_help()
        return 1

    try:
        results = frbench.download_assets(
            names=args.assets,
            download_all=args.all,
            force=args.force,
            refresh_manifest=args.refresh,
        )
    except frbench.FRBenchDownloadError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    ok = sum(results.values())
    print(f"\nFinished: {ok}/{len(results)} succeeded.")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
