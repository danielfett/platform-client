#!/usr/bin/env python3

"""
Copyright 2020 yes.com

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import argparse
import json
import sys
from base64 import b64decode
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from operator import attrgetter, itemgetter
from time import sleep
from typing import Dict, List, Optional, Set, Tuple

import requests
import requests_cache
from config_path import ConfigPath
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from rich import box
from rich.color import Color
from rich.console import Console, RenderGroup
from rich.panel import Panel
from rich.progress import Progress
from rich.style import Style
from rich.table import Table
from yaml import SafeLoader, load

nop = lambda x: x


PARALLEL_REQUESTS = 100
PARALLEL_REQUESTS_PLATFORM = 15
TIMEOUT = (14, 14)
CACHE_LIFETIME = 300

console = Console()


class YesPlatformAPI:
    DEFAULT_URLS = {
        "sandbox": {
            "token_endpoint": "https://as.sandbox.yes.com/token",
            "idps": "https://api.sandbox.yes.com/idps/v1/",
            "rps": "https://api.sandbox.yes.com/rps/v1/",
            "sps": "https://api.sandbox.yes.com/sps/v1/",
            "banks": "https://api.sandbox.yes.com/banks/v1/",
        },
        "production": {
            "token_endpoint": "https://as.yes.com/token",
            "idps": "https://api.yes.com/idps/v1/",
            "rps": "https://api.yes.com/rps/v1/",
            "sps": "https://api.yes.com/sps/v1/",
            "banks": "https://api.yes.com/banks/v1/",
        },
    }

    def __init__(self, client_id, cert, key, environment):
        self.client_id = client_id
        self.cert_pair = (cert, key)
        self.urls = self.DEFAULT_URLS[environment]
        self.get_token()

    def get_token(self):
        resp = requests.post(
            self.urls["token_endpoint"],
            data={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
            },
            cert=self.cert_pair,
            timeout=TIMEOUT,
        ).json()
        self.access_token = resp["access_token"]

    def get(self, endpoint, params={}):
        req = requests.get(
            self.urls[endpoint],
            params=params,
            headers={"Authorization": f"Bearer {self.access_token}"},
            cert=self.cert_pair,
            timeout=TIMEOUT,
        )
        return req.json()


def nice_formatter(inval):
    if inval is None:
        return ""
    elif isinstance(inval, CustomType):
        return inval.get_rich()
    elif type(inval) == list:
        return "[light_sky_blue1],[/light_sky_blue1] ".join(inval)
    else:
        return str(inval)


def raw_formatter(inval):
    if isinstance(inval, CustomType):
        return inval.raw
    return inval


def string_formatter(inval):
    return str(raw_formatter(inval))


class Dataset:
    rows: List[Dict]
    output_columns: Optional = None
    available_columns: Optional[Set] = None
    sort: Optional[Set] = None
    where: Optional[str] = None
    original_row_count: int = 0
    custom_types: Dict = {}

    STANDARD_API: Optional[str] = None

    def __init__(self, rows, sort=set()):
        self.rows = rows
        self.original_row_count = len(rows)
        self.available_columns = set()
        for row in self.rows:
            self.available_columns |= set(row.keys())
        self._sanitize()
        if not self.output_columns:
            self.output_columns = self.available_columns

    def _sanitize(self):
        for row in self.rows:
            for f in self.available_columns:
                if not f in row:
                    row[f] = None
                else:
                    if (
                        f in self.custom_types
                        and not type(row[f]) == self.custom_types[f]
                    ):
                        row[f] = self.custom_types[f](row[f])

    def limit_columns_to(self, columns):
        limit_columns = set(columns.split(","))
        self.output_columns = limit_columns.intersection(self.available_columns)

    def output_columns_append(self, columns):
        add_columns = tuple(columns.split(","))
        self.output_columns += add_columns

    def output_all_columns(self):
        self.output_columns = self.available_columns

    def set_sort(self, columns):
        self.sort = tuple(columns.split(","))

    def get_rows(self, fill_blanks, blank="", formatter=raw_formatter, sort=True):
        self._sort()
        for row in self._where():
            out = {}
            for col in self.output_columns:
                if col in row:
                    out[col] = formatter(row[col])
                elif fill_blanks:
                    out[col] = blank
                else:
                    pass
            yield out

    def get_rows_list(self, blank="", formatter=raw_formatter):
        self._sort()
        for row in self._where():
            yield [formatter(row.get(col, blank)) for col in self.output_columns]

    def _sort(self):
        if self.sort:
            self.rows.sort(key=itemgetter(*self.sort))

    def _where(self):
        if self.where is None:
            yield from self.rows
        else:
            for row in self.rows:
                try:
                    if eval(self.where, {}, row):
                        yield row
                except TypeError:
                    pass

    @classmethod
    def get(cls, api, args):
        with console.status("Retrieving list ..."):
            rows = api.get(cls.STANDARD_API)
        return cls(rows)


def retrieve_auxilliary_data_parallel(
    api, dataset: Dataset, parallel_requests, update_fn, new_dataset_class, status_text
):
    orig_data = dataset.rows
    data = []

    def progress_bar():
        with Progress() as progress:
            task = progress.add_task(status_text, total=len(orig_data))
            while len(data) < len(orig_data):
                progress.update(task, completed=len(data))
                sleep(0.1)

    with ThreadPoolExecutor(max_workers=parallel_requests) as executor:
        executor.submit(progress_bar)
        executor.map(lambda x: update_fn(data, api, x), orig_data)

    return new_dataset_class(data)


class CustomType:
    pass


class JWKS(CustomType):
    def __init__(self, jwks):
        self.raw = jwks
        self.certs = [self._read_certificate(c["x5c"][0]) for c in jwks["keys"]]
        self.min_not_valid_after = max(
            self.certs, key=attrgetter("not_valid_after")
        ).not_valid_after
        self.lifetime_days = (self.min_not_valid_after - datetime.now()).days

    def _read_certificate(self, pem):
        return x509.load_der_x509_certificate(b64decode(pem), default_backend())

    def get_rich(self):
        out = RenderGroup()

        for c in self.certs:
            try:
                if (c.not_valid_after - datetime.now()).days < 100:
                    color = "red"
                else:
                    color = "green"

                out.renderables.append(
                    Panel(
                        f"{c.subject.rfc4514_string()}\n[{color}]valid until {c.not_valid_after}[/{color}]"
                    )
                )
            except Exception as e:
                out.renderables.append(Panel(f"[red](unable to decode: {e})[/red]"))
        return out


class IDPDataset(Dataset):
    output_columns = ("id", "iss", "bics", "owner_id")
    sort = ("bics",)
    where = "status=='active'"
    STANDARD_API = "idps"


class IDPBankDataset(IDPDataset):
    output_columns = ("id", "iss", "name")

    @classmethod
    def get(cls, api, args):
        iss_dataset = super().get(api, args)
        return retrieve_auxilliary_data_parallel(
            api,
            iss_dataset,
            PARALLEL_REQUESTS_PLATFORM,
            fetch_and_store_bank,
            cls,
            "Fetching bank information",
        )


class IDPIssuerDataset(IDPDataset):
    @classmethod
    def get(cls, api, args):
        iss_dataset = super().get(api, args)
        return retrieve_auxilliary_data_parallel(
            api,
            iss_dataset,
            PARALLEL_REQUESTS,
            fetch_and_store_issuer,
            cls,
            "Fetching OIDC configuration files",
        )


class IDPIssuerBankDataset(IDPBankDataset):
    @classmethod
    def get(cls, api, args):
        iss_dataset = super().get(api, args)
        return retrieve_auxilliary_data_parallel(
            api,
            iss_dataset,
            PARALLEL_REQUESTS,
            fetch_and_store_issuer,
            cls,
            "Fetching OIDC configuration files",
        )


class RPDataset(Dataset):
    output_columns = ("client_id", "client_name")
    sort = ("client_id",)
    where = "status=='active'"
    custom_types = {"jwks": JWKS}
    STANDARD_API = "rps"


class SPDataset(Dataset):
    output_columns = ("client_id", "client_name")
    sort = ("client_id",)
    where = "status=='active'"
    custom_types = {"jwks": JWKS}
    STANDARD_API = "sps"


def update_and_remap_keys(existing_dict, prefix, dct):
    for key, value in dct.items():
        if key in existing_dict:
            existing_dict[f"{prefix}__{key}"] = value
        else:
            existing_dict[key] = value


def fetch_and_store_issuer(out_data, _, idp):
    ERROR_CONST = "__oidc_error"
    config_url = f"{idp['iss']}/.well-known/openid-configuration"
    try:
        config_file = requests.get(
            config_url,
            timeout=TIMEOUT,
        ).json()
    except requests.exceptions.ConnectionError:
        idp[ERROR_CONST] = "Failed to connect."
    except json.decoder.JSONDecodeError:
        idp[ERROR_CONST] = "Failed to parse OIDC config."
    except Exception as e:
        idp[ERROR_CONST] = repr(e)
    else:
        idp[ERROR_CONST] = None
        update_and_remap_keys(idp, "oidc", config_file)
    finally:
        out_data.append(idp)


def fetch_and_store_bank(out_data, api, idp):
    ERROR_CONST = "__bank_error"
    try:
        bank_information = api.get("banks", {"term": idp["bics"][0], "limit": 1})[0]
        update_and_remap_keys(idp, "bank", bank_information)
    except Exception as e:
        idp[ERROR_CONST] = e
    finally:
        out_data.append(idp)


def output_json_lines(data: Dataset, formatter, cache_disabled):
    for el in data.get_rows(False, sort=False):
        print(json.dumps(el))


def output_json_list(data: Dataset, formatter, cache_disabled):
    print(json.dumps(list(data.get_rows(False, sort=False))))


def output_rich(data: Dataset, formatter, cache_disabled):
    console.record = True
    table = Table(
        show_header=True,
        header_style="bold blue",
        row_styles=[Style(), Style(color=Color.from_ansi(252))],
    )
    table.box = box.MINIMAL

    for col in data.output_columns:
        table.add_column(col)

    for row in data.get_rows_list("", formatter=formatter):
        table.add_row(*row)

    console.print(table)
    console.rule("Information")

    if len(data.output_columns) < len(data.available_columns):
        console.print(
            "[green]More columns available:[/green]",
            ", ".join(sorted(data.available_columns - set(data.output_columns))),
        )
    console.print("[green]Sorted by:[/green]", ", ".join(data.sort))
    if data.where is None:
        console.print(
            f"[green]All available rows ({table.row_count}) are shown.[/green]"
        )
    else:
        console.print(
            "[green]Filter active:[/green]",
            f"`[blue]{data.where}[/blue]`",
            f"→ {table.row_count} rows of {data.original_row_count} available rows shown.",
        )
    if not cache_disabled:
        console.print("[green]Caching enabled. To disable, use[/green] --no-cache")


def filter_only(fields, expr):
    selected = set(expr.split(","))
    return set(fields).intersection(selected)


FORMAT_OPTS = {
    "json-lines": {"function": output_json_lines, "disable_cache": True},
    "json-list": {"function": output_json_list, "disable_cache": True},
    "table": {"function": output_rich, "disable_cache": False},
}

if __name__ == "__main__":
    configpath = ConfigPath("yes", "platform-client", "")
    requests_cache.core.install_cache(
        cache_name=str(configpath.saveFilePath(mkdir=True)),
        fast_save=True,
        expire_after=CACHE_LIFETIME,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("credentials_file", type=argparse.FileType("r"))

    subparsers = parser.add_subparsers()

    create_subparsers = [
        ("idps", IDPDataset),
        ("idps+oidc", IDPIssuerDataset),
        ("idps+banks", IDPBankDataset),
        ("idps+banks+oidc", IDPIssuerBankDataset),
        ("rps", RPDataset),
        ("sps", SPDataset),
    ]

    for sp_name, sp_func in create_subparsers:
        p = subparsers.add_parser(sp_name)
        p.set_defaults(func=sp_func.get)

        p.add_argument(
            "--no-cache",
            "-n",
            action="store_true",
            help=f"Disable caching. By default, all requests are cached for {CACHE_LIFETIME} seconds.",
        )

        p.add_argument(
            "--with",
            type=str,
            default=None,
            help="Add certain fields to output, provided as comma-separated list. E.g.: --with allowed_claims",
        )
        p.add_argument(
            "--with-all",
            action="store_true",
            help="Include all available fields in the output",
        )
        p.add_argument(
            "--only",
            type=str,
            default=None,
            help="Limit output to certain fields, provided as comma-separated list. E.g.: --only active,id",
        )
        p.add_argument(
            "--format",
            "-f",
            choices=FORMAT_OPTS.keys(),
            default="table",
            help="Define the output format.",
        )
        p.add_argument(
            "--export",
            "-e",
            type=str,
            help="Export to HTML file.",
        )
        p.add_argument(
            "--sort",
            type=str,
            default=None,
            help="Sort by field(s), provided as comma-separated list. E.g.: --sort name",
        )
        p.add_argument(
            "--where",
            type=str,
            default=None,
            help="Add/modify filter. Python expression, row field available as variables, e.g. --where '\"bankid\" in issuer_url'. Use --all to disable filtering. ",
        )
        p.add_argument(
            "--all-rows",
            action="store_true",
            help="Disable filtering. ",
        )
        p.add_argument(
            "--raw",
            action="store_true",
            help="Disable conversion of data to more readable representations. ",
        )

    args = parser.parse_args()
    if args.where:
        compile(args.where, "provided where expression", "eval")

    no_cache = args.no_cache or FORMAT_OPTS[args.format]["disable_cache"]
    if no_cache:
        requests_cache.core.clear()

    with console.status("Connecting to yes® platform ...") as status:
        api = YesPlatformAPI(**load(args.credentials_file.read(), Loader=SafeLoader))
    data: Dataset = args.func(api, args)
    if args.only:
        data.limit_columns_to(args.only)
    elif getattr(args, "with"):
        data.output_columns_append(getattr(args, "with"))
    if args.with_all:
        data.output_all_columns()
    if args.sort:
        data.set_sort(args.sort)
    if args.where:
        compile(args.where, "provided where expression", "eval")
        data.where = args.where
    elif args.all_rows:
        data.where = None
    FORMAT_OPTS[args.format]["function"](
        data,
        formatter=(string_formatter if args.raw else nice_formatter),
        cache_disabled=no_cache,
    )
    if args.export:
        console.save_html(args.export)
        console.print(f"[red]Output exported to {args.export}.")

    requests_cache.core.remove_expired_responses()
