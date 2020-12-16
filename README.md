# yes® Platform Client

**NOTE:** This software is *not* required to use yes® services. Do not rely on it unless explicitly asked to do so.

## Set-Up

```
git clone https://github.com/yescom/platform-client.git
cd platform-client
pip3 install -r requirements.txt
```

You need to acquire credentials to access the yes® platform. Create a configuration file containing these credentials, e.g., `sandbox.yml`, as follows:

```yaml
client_id: "pf:sandbox.yes.com:00000000-0000-0000-0000-00000000000"  # your platform client ID
cert: ".../platform/cert.pem"  # path to certificate file
key: ".../platform/key.pem"  # path to key file
environment: sandbox  # or "production"
```

See [credentials.example.yml](credentials.example.yml) for an example.

## Usage

List all issuers (`idps`), relying parties (`rps`), service providers (`sps`):

```bash
$ ./client.py sandbox.yml idps

  id                         │ iss                        │ bics                        │ owner_id    
 ────────────────────────────┼────────────────────────────┼─────────────────────────────┼────────────╴
  a77f5a6f-4eea-44b1-8c2f-c… │ https://issuer.example.de… │ ZZZZDEM1BGL                 │ xidp        
  ee1f68aa-433a-4b1d-b10c-a… │ https://69968840.meine-ba… │ XXYYDEF0018                 │ 420081      
  47768295-9122-4a82-a14f-a… │ https://69960088.meine-ba… │ XXYYDEF002S                 │ 420081      
  3cd4f668-a766-45d2-bedd-8… │ https://69968204.meine-ba… │ XXYYDEF0204                 │ 420081      
  9c7ccfd0-91de-4762-a513-a… │ https://69968805.meine-ba… │ XXYYDEF0805                 │ 420081      

More columns available: notification_email, notification_email_senders, status
Sorted by: bics
Filter active: `status=='active'` → 54 rows of 54 available rows shown.
```
**Note:** Default settings for columns, sorting, and filtering are applied. 

### Combining Datasets
The list of issuers (`idps`) can be combined with bank information (`+banks`) and/or with the information retrievable from the OpenID Connect configuration files. Valid combinations: `idps+banks`, `idps+oidc`, and `idps+banks+oidc`. Fields that occur in multiple datasets are prefixed with `bank__` and `oidc__`, respectively. When an issuer file or bank information cannot be retrieved, the special columns `OIDC_ERROR` and `BANK_ERROR` contains the respective error messages. 


### Show/Hide Columns

Add more colums/fields to the output (see list of additionally available columns below the table above):
```bash
$ ./client.py sandbox.yml idps --with status,notification_email
```
To include all columns in the output, use `--all-rows`.

Limit output to certain columns/fields:
```bash
$ ./client.py sandbox.yml idps --only status,id
```

### Sorting

Sort by one or more fields:
```bash
$ ./client.py sandbox.yml idps --sort owner_id
```

### Filtering

Limit output by providing a condition:
```bash
$ ./client.py sandbox.yml idps --where <condition>
```
`<condition>` must be a valid python expression. Fields are available as local variables. Examples:

 * Only active RPs:  \
    `$ ./client.py sandbox.yml rps --where 'status=="inactive"'`
 * RPs containing 'test' in their name: \
    `$ ./client.py sandbox.yml rps --where '"test" in client_name.lower()'`
 * IDPs where the ID starts with 'ff': \
    `$ ./client.py sandbox.yml idps --where 'id.startswith("ff")'`
 * Issuer information where an authorization endpoint is available: \
    `$ ./client.py sandbox.yml idps+oidc --where 'authorization_endpoint'`
 * Service providers supporting a particular conformance level: \
    `$ ./client.py sandbox.yml sps --with conformance_levels_supported --where '"AdES-B-LT" in conformance_levels_supported'`

(Note: `--where` conditions operate on the raw data. Some columns, like `conformance_levels_supported` are lists and are by default converted to more readable strings for output. Use `--raw` to see the data structures in the table output.)

Certificates stored in the `jwks` property of RPs and SPs have a special property `jwks.lifetime_days` that represents the lifetime of the certificate in the JWKS that expires the earliest. This can be used for filtering RPs and SPs with certificates expiring soon:

```bash
$ ./client.py sandbox.yml rps --with jwks --where 'jwks.lifetime_days < 100'
```

## JSON Dumping
Use `--format` to control the output of data:

* `--format json-list` outputs all rows as one JSON list of objects, each object representing a data row. 
* `--format json-lines` outputs one line for each data row, each containing an object in JSON notation.

Example: Show all authorization endpoint URLs.
```bash
$ ./client.py sandbox.yml idps+oidc --only authorization_endpoint -f json-lines --where 'authorization_endpoint'
{"authorization_endpoint": "https://example.com/oidc/auth"}
{"authorization_endpoint": "https://idp.other.example/yes"}
...
```
## HTML Export
Use `--export <filename>` to render the results to an HTML file.

