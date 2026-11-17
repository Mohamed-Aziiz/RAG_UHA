"""
diag_tls.py - Pourquoi Python refuse le certificat d'un site, et comment y remedier.

    uv run python scripts/diag_tls.py
    uv run python scripts/diag_tls.py --host www.uha.fr

Le script tente trois connexions et compare :
  A. magasin certifi   - ce qu'utilise httpx par defaut
  B. magasin du systeme - via truststore, c'est ce que fait ton navigateur
  C. magasin du systeme - via le module ssl seul (racines Windows, sans AIA)

Puis il lit le certificat presente par le serveur et nomme la cause probable.
Compatible Python 3.11. Aucune dependance obligatoire hors httpx.
"""

from __future__ import annotations

import argparse
import re
import socket
import ssl
import sys
import tempfile
from pathlib import Path

import httpx

try:
    import truststore
except ImportError:
    truststore = None

try:
    import certifi
except ImportError:
    certifi = None

# Racines installees localement par un antivirus, un pare-feu ou un reseau d'entreprise.
INTERCEPTORS = [
    "kaspersky",
    "eset",
    "avast",
    "avg",
    "bitdefender",
    "sophos",
    "mcafee",
    "norton",
    "symantec",
    "f-secure",
    "trend micro",
    "fortinet",
    "fortigate",
    "zscaler",
    "netskope",
    "bluecoat",
    "blue coat",
    "palo alto",
    "checkpoint",
    "check point",
    "cisco umbrella",
    "forcepoint",
    "sonicwall",
    "watchguard",
    "proxy",
    "firewall",
    "ssl inspection",
    "web filter",
]

# Autorites publiques courantes (les universites francaises passent par RENATER/GEANT).
PUBLIC_CAS = [
    "sectigo",
    "digicert",
    "let's encrypt",
    "letsencrypt",
    "globalsign",
    "geant",
    "terena",
    "usertrust",
    "comodo",
    "thawte",
    "entrust",
    "godaddy",
    "amazon",
    "google trust",
    "certigna",
    "actalis",
    "buypass",
    "isrg",
    "rapidssl",
    "ssl.com",
    "identrust",
]


def rdn_to_text(rdn) -> str:
    """Transforme la structure imbriquee du module ssl en texte lisible."""
    parts = []
    for item in rdn or ():
        for key, value in item:
            parts.append(f"{key}={value}")
    return ", ".join(parts)


def decode_cert(der: bytes) -> dict:
    """Decode un certificat DER. cryptography si disponible, sinon le module ssl."""
    try:
        from cryptography import x509

        cert = x509.load_der_x509_certificate(der)
        aia = []
        try:
            ext = cert.extensions.get_extension_for_class(x509.AuthorityInformationAccess).value
            aia = [
                d.access_location.value
                for d in ext
                if d.access_method == x509.oid.AuthorityInformationAccessOID.CA_ISSUERS
            ]
        except Exception:
            pass
        return {
            "subject": cert.subject.rfc4514_string(),
            "issuer": cert.issuer.rfc4514_string(),
            "not_after": cert.not_valid_after_utc.isoformat(timespec="seconds"),
            "ca_issuers": aia,
        }
    except ImportError:
        pass
    except Exception as exc:
        return {"error": f"decodage impossible : {exc}"}

    # Repli stdlib : passer par un fichier PEM temporaire.
    try:
        pem = ssl.DER_cert_to_PEM_cert(der)
        with tempfile.NamedTemporaryFile("w", suffix=".pem", delete=False, encoding="ascii") as fh:
            fh.write(pem)
            tmp = fh.name
        try:
            info = ssl._ssl._test_decode_cert(tmp)  # API privee mais stable
        finally:
            Path(tmp).unlink(missing_ok=True)
        return {
            "subject": rdn_to_text(info.get("subject")),
            "issuer": rdn_to_text(info.get("issuer")),
            "not_after": str(info.get("notAfter", "?")),
            "ca_issuers": list(info.get("caIssuers") or ()),
        }
    except Exception as exc:
        return {"error": f"decodage impossible : {exc}"}


def peer_cert_der(host: str, port: int, timeout: float) -> bytes | None:
    """Recupere le certificat presente, sans le valider (inspection seule)."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    try:
        with (
            socket.create_connection((host, port), timeout=timeout) as sock,
            ctx.wrap_socket(sock, server_hostname=host) as tls,
        ):
            return tls.getpeercert(binary_form=True)
    except Exception as exc:
        print(f"  connexion impossible meme sans verification : {exc}")
        return None


def chain_via_openssl(host: str, port: int) -> list[str]:
    """Chaine complete envoyee par le serveur, si openssl est disponible."""
    import subprocess

    try:
        proc = subprocess.run(
            [
                "openssl",
                "s_client",
                "-connect",
                f"{host}:{port}",
                "-servername",
                host,
                "-showcerts",
            ],
            input=b"",
            capture_output=True,
            timeout=20,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return []
    text = proc.stdout.decode("utf-8", "replace") + proc.stderr.decode("utf-8", "replace")
    return re.findall(r"^\s*\d+\s+s:(.+)$", text, re.MULTILINE)


def try_request(label: str, url: str, verify, timeout: float) -> bool:
    print(f"  {label:<42} ", end="", flush=True)
    try:
        with httpx.Client(verify=verify, timeout=timeout, follow_redirects=True) as client:
            resp = client.get(url)
        print(f"OK (HTTP {resp.status_code})")
        return True
    except ssl.SSLError as exc:
        print(f"ECHEC TLS : {getattr(exc, 'reason', '') or exc}")
    except httpx.ConnectError as exc:
        msg = str(exc)
        label = "ECHEC TLS" if "CERTIFICATE" in msg.upper() else "ECHEC CONNEXION"
        print(f"{label} : {msg[:100]}")
    except Exception as exc:
        print(f"ECHEC : {type(exc).__name__} {str(exc)[:70]}")
    return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Diagnostic TLS pour le scraping.")
    ap.add_argument("--host", default="www.uha.fr")
    ap.add_argument("--port", type=int, default=443)
    ap.add_argument("--path", default="/robots.txt")
    ap.add_argument("--timeout", type=float, default=20.0)
    args = ap.parse_args(argv)

    url = f"https://{args.host}:{args.port}{args.path}"
    print(f"Diagnostic TLS : {url}")
    print(f"Python {sys.version.split()[0]} | httpx {httpx.__version__}")
    print(f"certifi : {certifi.where() if certifi else 'absent'}")
    print(f"truststore : {'installe' if truststore else 'ABSENT'}")
    print()

    # ---- Les trois tentatives -------------------------------------------
    print("Tentatives de connexion")
    ok_certifi = try_request("A. certifi (defaut de httpx)", url, True, args.timeout)

    ok_truststore = False
    if truststore is not None:
        ctx = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ok_truststore = try_request("B. magasin systeme (truststore)", url, ctx, args.timeout)
    else:
        print("  B. magasin systeme (truststore)          non teste : truststore absent")

    ctx_std = ssl.create_default_context()
    ok_stdlib = try_request("C. magasin systeme (module ssl)", url, ctx_std, args.timeout)
    print()

    # ---- Le certificat presente ------------------------------------------
    print("Certificat presente par le serveur")
    der = peer_cert_der(args.host, args.port, args.timeout)
    issuer = ""
    if der:
        info = decode_cert(der)
        if "error" in info:
            print(f"  {info['error']}")
        else:
            issuer = info["issuer"]
            print(f"  sujet    : {info['subject']}")
            print(f"  emetteur : {issuer}")
            print(f"  expire   : {info['not_after']}")
            if info["ca_issuers"]:
                print(f"  AIA      : {info['ca_issuers'][0]}")
            else:
                print(
                    "  AIA      : absent (le certificat n'indique pas ou trouver l'intermediaire)"
                )
    chain = chain_via_openssl(args.host, args.port)
    if chain:
        print(f"  chaine envoyee : {len(chain)} certificat(s)")
        for i, subject in enumerate(chain):
            print(f"    [{i}] {subject.strip()[:95]}")
        if len(chain) == 1:
            print("  -> le serveur n'envoie que le certificat final, sans intermediaire")
    else:
        print("  chaine complete : openssl indisponible, inspection limitee au certificat final")
    print()

    # ---- Verdict ----------------------------------------------------------
    low = issuer.lower()
    intercepted = next((n for n in INTERCEPTORS if n in low), None)
    public_ca = next((n for n in PUBLIC_CAS if n in low), None)

    print("=" * 68)
    if ok_certifi:
        print("VERDICT : aucun probleme TLS depuis cette machine.")
        print("Le scraper fonctionne tel quel. Si tu vois encore l'erreur, c'est")
        print("qu'une autre machine ou un autre reseau est en cause.")
        return 0

    if intercepted:
        print(f"VERDICT : interception TLS locale detectee ({intercepted}).")
        print("Un antivirus, un pare-feu ou le reseau de l'universite remplace le")
        print("certificat du site par le sien. Sa racine est installee dans Windows,")
        print("mais pas dans certifi : d'ou l'echec de httpx alors que le navigateur passe.")
    elif public_ca and chain and len(chain) == 1:
        print(f"VERDICT : chaine incomplete cote serveur (emetteur : {public_ca}).")
        print("Le serveur n'envoie pas le certificat intermediaire. Les navigateurs le")
        print("telechargent tout seuls (AIA), pas certifi.")
    elif public_ca:
        print(f"VERDICT : autorite publique ({public_ca}) non validee par certifi.")
        print("Soit l'intermediaire manque, soit certifi est trop ancien.")
    else:
        print("VERDICT : emetteur non reconnu comme autorite publique connue.")
        print("Probablement une racine locale (entreprise, universite, antivirus).")
    print()

    print("QUE FAIRE")
    if ok_truststore:
        print("  Le magasin de certificats du systeme valide ce site (test B reussi),")
        print("  et truststore est deja installe. Le scraper l'utilise automatiquement")
        print("  (option --ssl auto, par defaut). Il n'y a rien a installer : relance")
        print("  simplement ta commande.")
        print()
        print("      uv run python scripts/scrape_uha.py --dry-run")
        print()
        print("  Tu dois voir en premiere ligne :")
        print("      [tls] magasin de certificats du systeme (truststore)")
        print()
        print("  A noter pour ton rapport : la chaine incomplete est un defaut de")
        print("  configuration du serveur, pas de ta machine. truststore fonctionne")
        print("  parce que Windows telecharge l'intermediaire via l'URL AIA. Sous")
        print("  Linux, OpenSSL ne le fait pas : la, il faudrait --ca-bundle avec")
        print("  l'intermediaire ajoute au paquet certifi.")
    elif truststore is None:
        print("  truststore n'est pas installe. C'est le remede le plus probable :")
        print("  il fait valider le certificat par Windows, comme ton navigateur.")
        print()
        print("      uv add truststore")
        print("      uv run python scripts/diag_tls.py      # relance ce diagnostic")
        print()
    elif ok_stdlib:
        print("  Le module ssl seul fonctionne. Lance le scraper avec :")
        print()
        print("      uv run python scripts/scrape_uha.py --ssl system --dry-run")
        print()
    else:
        print("  Ni certifi ni le magasin Windows ne valident ce certificat.")
        print("  Exporte la chaine depuis ton navigateur (cadenas > certificat >")
        print("  details > exporter, format PEM, chaine complete) puis :")
        print()
        print(
            "      uv run python scripts/scrape_uha.py --ca-bundle C:\\chemin\\chaine.pem --dry-run"
        )
        print()
        print("  Si meme cela echoue, le reseau que tu utilises bloque peut-etre le site :")
        print("  essaie depuis une autre connexion (partage de connexion du telephone).")
    print("=" * 68)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
