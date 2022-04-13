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
import csv
import json
import sys
from base64 import b64decode
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from io import StringIO
from operator import attrgetter, itemgetter
from time import sleep
from typing import Dict, List, Optional, Set
import logging
from cryptography.hazmat.primitives import hashes

import dateparser
import requests
import requests_cache
from config_path import ConfigPath
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from rich import box
from rich.color import Color
from rich.console import Console, RenderGroup
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress
from rich.style import Style
from rich.syntax import Syntax
from rich.table import Table
from yaml import SafeLoader, dump, load

nop = lambda x: x


PARALLEL_REQUESTS = 100
PARALLEL_REQUESTS_PLATFORM = 15
TIMEOUT = (14, 14)
CACHE_LIFETIME = 600

console = Console()


class TooManyResultsException(Exception):
    pass


class YesPlatformAPI:
    DEFAULT_URLS = {
        "sandbox": {
            "token_endpoint": "https://as.sandbox.yes.com/token",
            "idps": "https://api.sandbox.yes.com/idps/v1/",
            "rps": "https://api.sandbox.yes.com/rps/v1/",
            "sps": "https://api.sandbox.yes.com/sps/v1/",
            "banks": "https://api.sandbox.yes.com/banks/v1/",
            "mrs": "https://api.sandbox.yes.com/mediationrecords/v2/",
            "sc": "https://api.sandbox.yes.com/service-configuration/v1/",
            "pfclients": "https://as.sandbox.yes.com/clients",
        },
        "production": {
            "token_endpoint": "https://as.yes.com/token",
            "idps": "https://api.yes.com/idps/v1/",
            "rps": "https://api.yes.com/rps/v1/",
            "sps": "https://api.yes.com/sps/v1/",
            "banks": "https://api.yes.com/banks/v1/",
            "mrs": "https://api.yes.com/mediationrecords/v2/",
            "sc": "https://api.yes.com/service-configuration/v1/",
            "pfclients": "https://as.sandbox.yes.com/clients",
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
        resp = req.json()
        if "error" in resp:
            if resp["error"] == "invalid_request" and resp[
                "error_description"
            ].startswith("There are more results"):
                raise TooManyResultsException()
            else:
                raise Exception(resp["error_description"])
        return resp


class Dataset:
    rows: List[Dict]
    output_columns: Optional[List] = None
    available_columns: Optional[Set] = None
    sort: Optional[Set] = None
    where: Optional[str] = None
    original_row_count: int = 0
    custom_types: Dict = {}
    add_ons: List = []

    CLI_IDENTIFIER: str
    CLI_QUICK_SEARCH: List[str] = []
    STANDARD_API: Optional[str] = None

    def __init__(self, rows, sort=set()):
        self.rows = rows
        self._enrich()
        self.original_row_count = len(rows)
        self.available_columns = set()
        for row in self.rows:
            self.available_columns |= set(row.keys())
        self._sanitize()
        if not self.output_columns:
            self.output_columns = list(self.available_columns)

    def _enrich(self):
        pass

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
        limit_columns = columns.split(",")
        self.output_columns = list(
            x for x in limit_columns if x in self.available_columns
        )

    def output_columns_append(self, columns):
        for c in columns.split(","):
            if c in self.available_columns and not c in self.output_columns:
                self.output_columns.append(c)

    def output_all_columns(self):
        self.output_columns = self.available_columns

    def set_sort(self, columns):
        self.sort = tuple(columns.split(","))

    def get_rows(self, formatter, fill_blanks, blank="", sort=True):
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

    def get_rows_list(self, formatter, blank=""):
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
                # try:
                if eval(self.where, {}, row):
                    yield row
            # except (TypeError, AttributeError):
            #    pass

    def opportunistic_where_applied(self):
        if self.where is None:
            yield from self.rows
        else:
            for row in self.rows:
                try:
                    if eval(self.where, {}, row):
                        yield row
                except (TypeError, NameError):
                    yield row

    @classmethod
    def get(cls, api: YesPlatformAPI, args):
        with console.status(f"Retrieving list ({cls.CLI_IDENTIFIER})..."):
            rows = api.get(cls.STANDARD_API)

        data = cls(rows)
        # data.handle_args(args)

        selected_add_ons_names = args.action_name.split("+")
        selected_add_ons = []
        for addon in cls.add_ons:
            if addon.CLI_IDENTIFIER in selected_add_ons_names:
                data = cls(addon.enrich(api, data))
                selected_add_ons.append(addon)

        data.handle_args(args, selected_add_ons)
        return data

    def where_from_search(self, search, selected_add_ons):
        new_where = []
        for cls in selected_add_ons + [self]:
            for f in cls.CLI_QUICK_SEARCH:
                new_where.append(f"{repr(search.lower())} in {f}.lower()")

        if self.where:
            self.where = f"({self.where}) and ({' or '.join(new_where)})"
        else:
            self.where = " or ".join(new_where)

    def handle_args(self, args, selected_add_ons=[]):
        if args.where:
            compile(args.where, "provided where expression", "eval")
        if args.only:
            self.limit_columns_to(args.only)
        elif getattr(args, "with"):
            self.output_columns_append(getattr(args, "with"))
        if args.with_all:
            self.output_all_columns()
        if args.sort:
            self.set_sort(args.sort)

        if args.where:
            self.where = args.where
        elif args.all_rows:
            self.where = None
        if self.CLI_QUICK_SEARCH and args.search:
            self.where_from_search(args.search, selected_add_ons)

    @classmethod
    def add_subparser(cls, subparsers):
        aliases = [cls.CLI_IDENTIFIER]
        for addon in cls.add_ons:
            for el in list(aliases):
                aliases.append(f"{el}+{addon.CLI_IDENTIFIER}")

        p = subparsers.add_parser(aliases[0], aliases=aliases[1:])
        p.set_defaults(sp_class=cls)

        if cls.CLI_QUICK_SEARCH:
            p.add_argument(
                "search",
                nargs="?",
                default=None,
                help="Quick search in a selection of fields.",
            )

        group = p.add_mutually_exclusive_group()

        group.add_argument(
            "--with",
            type=str,
            default=None,
            help="Add certain fields to output, provided as comma-separated list. E.g.: --with allowed_claims",
        )
        group.add_argument(
            "--with-all",
            action="store_true",
            help="Include all available fields in the output",
        )
        group.add_argument(
            "--only",
            type=str,
            default=None,
            help="Limit output to certain fields, provided as comma-separated list. E.g.: --only active,id",
        )
        p.add_argument(
            "--sort",
            type=str,
            default=None,
            help="Sort by field(s), provided as comma-separated list. E.g.: --sort name",
        )

        group = p.add_mutually_exclusive_group()
        group.add_argument(
            "--where",
            type=str,
            default=None,
            help="Add/modify filter. Python expression, row field available as variables, e.g. --where '\"bankid\" in issuer_url'. Use --all to disable filtering. ",
        )
        group.add_argument(
            "--all-rows",
            action="store_true",
            help="Disable filtering. ",
        )
        # p.add_argument(
        #    "--raw",
        #    action="store_true",
        #    help="Disable conversion of data to more readable representations. ",
        # )
        return p


class CustomType:
    raw: object


class JWKSCustomType(CustomType):
    RED_THRESHOLD_DAYS = 30

    def __init__(self, jwks):
        self.raw = jwks
        self.certs = [self._read_certificate(c["x5c"][0]) for c in jwks["keys"]]
        try:
            self.min_not_valid_after = max(
                (c for c in self.certs if c is not None),
                key=attrgetter("not_valid_after"),
            ).not_valid_after
        except ValueError:
            self.min_not_valid_after = datetime.now()
        self.lifetime_days = (self.min_not_valid_after - datetime.now()).days

    def _read_certificate(self, pem):
        try:
            return x509.load_der_x509_certificate(b64decode(pem), default_backend())
        except Exception as e:
            return None

    def get_rich(self):
        out = RenderGroup()

        for c in self.certs:
            if c is None:
                out.renderables.append(
                    Panel(
                        "[red]Certificate could not be read. Please check the format.[/red]"
                    )
                )
                continue

            try:
                if (c.not_valid_after - datetime.now()).days < self.RED_THRESHOLD_DAYS:
                    color = "red"
                else:
                    color = "green"

                fp = c.fingerprint(hashes.SHA1()).hex(":").upper()

                out.renderables.append(
                    f"{c.subject.rfc4514_string()}\n  [{color}]valid until {c.not_valid_after}[/{color}]\n  {fp}"
                )
            except Exception as e:
                out.renderables.append(Panel(f"[red](unable to decode: {e})[/red]"))
        return out


class RemoteSignatureCreationCustomType(CustomType):
    def __init__(self, raw):
        self.raw = raw

    def get_rich(self):
        lines = []
        for endpoint in self.raw:
            out = f" * {endpoint['qtsp_id']} @ {endpoint['signDoc']}\n   conformance_levels_supported: {', '.join(endpoint['conformance_levels_supported'])}"
            lines.append(out)

        return Markdown("\n".join(lines))


class ClaimsCustomType(CustomType):
    def __init__(self, data):
        self.raw = data
        yml = StringIO()
        dump(self.raw, yml)
        self.yaml = yml.getvalue()

    def get_rich(self):
        return Syntax(
            self.yaml,
            "yaml",
            theme="ansi_dark",
        )

    def __contains__(self, key):
        return key in self.yaml


class DataAddOn:
    CLI_IDENTIFIER: str
    CLI_QUICK_SEARCH: List[str] = []
    OUTPUT_PREFIX: str
    ERROR_CONST: str
    IS_PLATFORM: True
    STATUS_TEXT: str

    def __init__(self):
        self.ERROR_CONST = f"__{self.OUTPUT_PREFIX}_error"

    def _update_and_remap_keys(self, existing_dict, dct, existing_columns):
        for key, value in dct.items():
            if key in existing_columns:
                existing_dict[f"{self.OUTPUT_PREFIX}__{key}"] = value
            else:
                existing_dict[key] = value

    @classmethod
    def enrich(cls, api, dataset: Dataset):
        orig_data = list(dataset.opportunistic_where_applied())
        data = []

        instance = cls()

        existing_columns = dataset.available_columns

        def progress_bar():
            with Progress() as progress:
                task = progress.add_task(instance.STATUS_TEXT, total=len(orig_data))
                while len(data) < len(orig_data):
                    progress.update(task, completed=len(data))
                    sleep(0.1)

        with ThreadPoolExecutor(
            max_workers=PARALLEL_REQUESTS_PLATFORM
            if cls.IS_PLATFORM
            else PARALLEL_REQUESTS
        ) as executor:
            executor.submit(progress_bar)
            executor.map(
                lambda x: instance.fetch_and_store(data, api, x, existing_columns),
                orig_data,
            )

        return data


class BankAddOn(DataAddOn):
    CLI_IDENTIFIER = "banks"
    CLI_QUICK_SEARCH = ["city", "bic"]
    OUTPUT_PREFIX = "bank"
    IS_PLATFORM = True
    STATUS_TEXT = "Retrieving bank information"

    def fetch_and_store(self, out_data, api: YesPlatformAPI, idp, existing_columns):
        try:
            bank_information = api.get("banks", {"term": idp["bics"][0], "limit": 1})[0]
            self._update_and_remap_keys(idp, bank_information, existing_columns)
        except Exception as e:
            idp[self.ERROR_CONST] = e
        finally:
            out_data.append(idp)


class OIDCAddOn(DataAddOn):
    CLI_IDENTIFIER = "oidc"
    OUTPUT_PREFIX = "oidc"
    IS_PLATFORM = False
    STATUS_TEXT = "Retrieving OIDC information"

    def fetch_and_store(self, out_data, _, idp, existing_columns):
        config_url = f"{idp['iss']}/.well-known/openid-configuration"
        try:
            config_file = requests.get(
                config_url,
                timeout=TIMEOUT,
            ).json()
        except requests.exceptions.ConnectionError:
            idp[self.ERROR_CONST] = "Failed to connect."
        except json.decoder.JSONDecodeError:
            idp[self.ERROR_CONST] = "Failed to parse OIDC config."
        except Exception as e:
            idp[self.ERROR_CONST] = repr(e)
        else:
            idp[self.ERROR_CONST] = None
            self._update_and_remap_keys(idp, config_file, existing_columns)
        finally:
            out_data.append(idp)


class SCAddOn(DataAddOn):
    CLI_IDENTIFIER = "sc"
    OUTPUT_PREFIX = "sc"
    IS_PLATFORM = True
    STATUS_TEXT = "Retrieving Service Configuration information"

    def fetch_and_store(self, out_data, api: YesPlatformAPI, idp, existing_columns):
        if idp["status"] == "inactive":
            idp[self.ERROR_CONST] = f"The issuer {idp['iss']} is not active."
            out_data.append(idp)
            return

        try:
            sc_information = api.get("sc", {"iss": idp["iss"]})
            self._update_and_remap_keys(idp, sc_information, existing_columns)
        except Exception as e:
            idp[self.ERROR_CONST] = e
        finally:
            out_data.append(idp)


class IDPDataset(Dataset):
    output_columns = ["id", "iss", "bics", "owner_id"]
    sort = ("bics",)
    where = "status=='active'"
    add_ons = [BankAddOn, OIDCAddOn, SCAddOn]
    custom_types = {"remote_signature_creation": RemoteSignatureCreationCustomType}

    STANDARD_API = "idps"
    CLI_IDENTIFIER = "idps"
    CLI_QUICK_SEARCH = ["id", "iss"]


class RPDataset(Dataset):
    output_columns = ["client_id", "client_name"]
    sort = ("client_id",)
    where = "status=='active'"
    custom_types = {"jwks": JWKSCustomType}

    STANDARD_API = "rps"
    CLI_IDENTIFIER = "rps"
    CLI_QUICK_SEARCH = ["client_id", "owner_id", "ac_redirect_uri", "client_name"]
    EDIT_LINK_FORMAT = (
        "https://partner.yes.com/relying-parties/update/{owner_id}/{client_id}"
    )

    def _enrich(self):
        for row in self.rows:
            row["edit_link"] = self.EDIT_LINK_FORMAT.format(**row)


class SPDataset(Dataset):
    output_columns = ["client_id", "client_name"]
    sort = ("client_id",)
    where = "status=='active'"
    custom_types = {"jwks": JWKSCustomType, "required_claims": ClaimsCustomType}

    STANDARD_API = "sps"
    CLI_IDENTIFIER = "sps"


class PFClientDataset(Dataset):
    output_columns = ["client_id", "client_name"]
    sort = ("client_id",)
    where = "status=='active'"
    custom_types = {"jwks": JWKSCustomType}

    STANDARD_API = "pfclients"
    CLI_IDENTIFIER = "clients"


class MRDataset(Dataset):
    output_columns = ["type", "client_id", "issuer"]
    sort = ("creation_time",)
    custom_types = {
        "requested_claims": ClaimsCustomType,
    }

    STANDARD_API = "mrs"
    CLI_IDENTIFIER = "mrs"
    RECURSION_LIMIT = 20
    DATE_FORMAT = "%Y-%m-%dT%H:%M:%S+00:00"

    @classmethod
    def add_subparser(cls, api):
        p = super().add_subparser(api)
        p.add_argument(
            "--from",
            "-f",
            required=True,
            help="Retrieve mediation records from this date/time. Relative times allowed, e.g., '1 month ago'.",
        )

        p.add_argument(
            "--to",
            "-t",
            help="Retrieve mediation records from this date/time. Relative times allowed, e.g., '1 month ago'.",
        )

        p.add_argument(
            "--owner-id",
            "-o",
            help="Retrieve mediation records only for this specific owner id.",
        )

    @classmethod
    def _update_and_remap_keys(cls, existing_dict, dct, existing_columns):
        for key, value in dct.items():
            if key in existing_columns:
                existing_dict[f"client__{key}"] = value
            else:
                existing_dict[key] = value

    @classmethod
    def get(cls, api: YesPlatformAPI, args):
        from_ = dateparser.parse(getattr(args, "from"), settings={"TO_TIMEZONE": "UTC"})

        if getattr(args, "to"):
            to_ = dateparser.parse(getattr(args, "to"), settings={"TO_TIMEZONE": "UTC"})
        else:
            to_ = datetime.now()

        if getattr(args, "owner_id"):
            owner_id = getattr(args, "owner_id")
        else:
            owner_id = None

        tracker = {"total_requests": 1, "completed_requests": 0}
        try:
            with console.status(f"Retrieving list ({cls.CLI_IDENTIFIER})..."):
                rows = list(cls._get_from_to(api, from_, to_, owner_id, tracker))
        except TooManyResultsException:
            console.print(
                "[red]Too many results, please define a shorter time window![/red]"
            )
            sys.exit(1)

        with console.status(f"Retrieving list ({RPDataset.CLI_IDENTIFIER})..."):
            rp_data = api.get(RPDataset.STANDARD_API)

        rp_data_by_client_id = {row["client_id"]: row for row in rp_data}

        data = cls(rows)
        # for row in data.rows:
        #    try:
        #        rp_entry = rp_data_by_client_id[row["client_id"]]
        #    except KeyError:
        #        rp_entry = {"__client_error": "Client ID not found."}
        #    cls._update_and_remap_keys(row, rp_entry, data.available_columns)
        data = cls(data.rows)
        data.handle_args(args)
        return data

    @classmethod
    def _get_from_to(
        cls,
        api: YesPlatformAPI,
        from_: datetime,
        to_: datetime,
        owner_id: Optional[str],
        tracker,
    ):
        assert from_ < to_
        if tracker["total_requests"] > cls.RECURSION_LIMIT:
            raise TooManyResultsException()

        data = {
            "from": from_.strftime(cls.DATE_FORMAT),
            "to": to_.strftime(cls.DATE_FORMAT),
        }

        if owner_id:
            data["owner_id"] = owner_id

        try:
            yield from api.get(
                cls.STANDARD_API,
                data,
            )
            tracker["completed_requests"] += 1
        except TooManyResultsException:
            middle = from_ + (to_ - from_) / 2
            # console.log(f"Recursing ({from_} → {middle} → {to_})")
            tracker["total_requests"] += 1
            yield from cls._get_from_to(api, from_, middle, owner_id, tracker)
            tracker["total_requests"] += 1
            yield from cls._get_from_to(api, middle, to_, owner_id, tracker)


def output_json_lines(data: Dataset, formatter, cache_disabled):
    for el in data.get_rows(formatter, False, sort=False):
        print(json.dumps(el))


def output_json_list(data: Dataset, formatter, cache_disabled):
    print(json.dumps(list(data.get_rows(formatter, False, sort=False))))


def output_rich(data: Dataset, formatter, cache_disabled):
    console.record = True
    table = Table(
        show_header=True,
        header_style="bold blue",
        row_styles=[Style(), Style(color=Color.from_ansi(252))],
        show_footer=True,
        footer_style="bold blue",
    )
    table.box = box.MINIMAL

    rows = list(data.get_rows_list(formatter))

    for col in data.output_columns:
        table.add_column(col, footer="" if len(rows) < 15 else col)

    for row in rows:
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


def output_csv(data: Dataset, formatter, cache_disabled):
    writer = csv.DictWriter(sys.stdout, fieldnames=data.output_columns)
    writer.writeheader()
    for row in data.get_rows(formatter, True):
        writer.writerow(row)


def filter_only(fields, expr):
    selected = set(expr.split(","))
    return set(fields).intersection(selected)


def rich_formatter(inval):
    if inval is None:
        return ""
    elif isinstance(inval, CustomType):
        return inval.get_rich()
    elif type(inval) == list:
        return "[light_sky_blue1],[/light_sky_blue1] ".join(str(x) for x in inval)
    else:
        return str(inval)


def raw_formatter(inval):
    if isinstance(inval, CustomType):
        return inval.raw
    return inval


def text_formatter(inval):
    if inval is None:
        return ""
    elif type(inval) == list:
        return ", ".join(str(x) for x in inval)
    return str(raw_formatter(inval))


FORMAT_OPTS = {
    "json-lines": {
        "function": output_json_lines,
        "formatter": raw_formatter,
        "disable_cache": False,
    },
    "json-list": {
        "function": output_json_list,
        "formatter": raw_formatter,
        "disable_cache": False,
    },
    "table": {
        "function": output_rich,
        "formatter": rich_formatter,
        "disable_cache": False,
    },
    "csv": {
        "function": output_csv,
        "formatter": text_formatter,
        "disable_cache": False,
    },
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
    parser.add_argument(
        "--no-cache",
        "-n",
        action="store_true",
        help=f"Disable caching. By default, all requests are cached for {CACHE_LIFETIME} seconds.",
    )
    parser.add_argument(
        "--format",
        choices=FORMAT_OPTS.keys(),
        default="table",
        help="Define the output format.",
    )
    parser.add_argument(
        "--export",
        type=str,
        help="Export to HTML file.",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging.",
    )
    subparsers = parser.add_subparsers(required=True, dest="action_name")

    create_subparsers = [
        IDPDataset,
        RPDataset,
        SPDataset,
        MRDataset,
        PFClientDataset,
    ]

    for sp_class in create_subparsers:
        sp_class.add_subparser(subparsers)

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    no_cache = args.no_cache or FORMAT_OPTS[args.format]["disable_cache"]
    if no_cache:
        requests_cache.core.clear()

    with console.status("Connecting to yes® platform ...") as status:
        api = YesPlatformAPI(**load(args.credentials_file.read(), Loader=SafeLoader))

    data: Dataset = args.sp_class.get(api, args)

    FORMAT_OPTS[args.format]["function"](
        data,
        formatter=FORMAT_OPTS[args.format]["formatter"],
        cache_disabled=no_cache,
    )
    if args.export:
        console.save_html(args.export)
        console.print(f"[red]Output exported to {args.export}.")

    requests_cache.core.remove_expired_responses()
