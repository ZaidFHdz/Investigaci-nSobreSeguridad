import argparse
import os
import sys
import warnings
from pathlib import Path

try:
    import tomllib
except ImportError:  # Python < 3.11
    tomllib = None

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        import google.generativeai as genai
except ImportError:
    genai = None


SECRETS_PATH = Path(".streamlit/secrets.toml")
SECRETS_TEMPLATE = """# Archivo local. No se sube a Git.
# Pega tu API key real entre comillas.
GEMINI_API_KEY = "TU_API_KEY_AQUI"

# Modelo recomendado para análisis más complejo del dashboard.
GEMINI_MODEL = "gemini-2.5-pro"
"""


def mensaje_configuracion_faltante():
    return f"""
No encontré GEMINI_API_KEY.

Opción recomendada para este proyecto:
1. Crea el archivo local:
   .streamlit/secrets.toml

2. Pon este contenido:
   GEMINI_API_KEY = "tu_api_key_real"
   GEMINI_MODEL = "gemini-2.5-pro"

También puedes generarlo con plantilla:
   .venv-general/bin/python listar_modelos_gemini.py --crear-secrets

Luego vuelve a correr:
   .venv-general/bin/python listar_modelos_gemini.py

Nota: .streamlit/secrets.toml está ignorado por Git para no subir tu key.
""".strip()


def crear_secrets_local():
    SECRETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if SECRETS_PATH.exists():
        print(f"Ya existe {SECRETS_PATH}. No lo sobrescribí.")
        return

    SECRETS_PATH.write_text(SECRETS_TEMPLATE, encoding="utf-8")
    print(f"Creé {SECRETS_PATH}. Abre el archivo y reemplaza TU_API_KEY_AQUI.")


def cargar_secret_streamlit(nombre):
    if tomllib is None or not SECRETS_PATH.exists():
        return ""

    try:
        with SECRETS_PATH.open("rb") as archivo:
            secrets = tomllib.load(archivo)
    except (OSError, tomllib.TOMLDecodeError):
        return ""

    return str(secrets.get(nombre, "") or "").strip()


def obtener_configuracion():
    api_key = (
        os.environ.get("GEMINI_API_KEY", "").strip()
        or cargar_secret_streamlit("GEMINI_API_KEY")
    )
    modelo_preferido = (
        os.environ.get("GEMINI_MODEL", "").strip()
        or cargar_secret_streamlit("GEMINI_MODEL")
        or "gemini-2.5-pro"
    )
    return api_key, modelo_preferido


def listar_modelos(solo_generate_content=True):
    if genai is None:
        raise RuntimeError(
            "Falta instalar `google-generativeai` en este entorno. "
            "Instala requirements.txt o usa el venv correcto."
        )

    api_key, modelo_preferido = obtener_configuracion()
    if not api_key:
        raise RuntimeError(mensaje_configuracion_faltante())

    genai.configure(api_key=api_key)

    print(f"Modelo preferido configurado: {modelo_preferido}")
    print("Modelos disponibles:")
    print()

    encontrados = 0
    for modelo in genai.list_models():
        metodos = list(getattr(modelo, "supported_generation_methods", []) or [])
        if solo_generate_content and "generateContent" not in metodos:
            continue

        encontrados += 1
        print(modelo.name)
        print("  Métodos:", ", ".join(metodos) if metodos else "No reportados")
        print()

    if encontrados == 0:
        print("No encontré modelos compatibles con el filtro actual.")


def probar_modelo(nombre_modelo):
    if genai is None:
        raise RuntimeError("Falta instalar `google-generativeai`.")

    api_key, _ = obtener_configuracion()
    if not api_key:
        raise RuntimeError(mensaje_configuracion_faltante())

    genai.configure(api_key=api_key)
    modelo = genai.GenerativeModel(nombre_modelo)
    respuesta = modelo.generate_content(
        "Responde solo con una frase breve: modelo listo."
    )
    print(getattr(respuesta, "text", "").strip() or "El modelo respondió sin texto.")


def main():
    parser = argparse.ArgumentParser(
        description="Lista modelos Gemini disponibles para la API key configurada."
    )
    parser.add_argument(
        "--todos",
        action="store_true",
        help="Muestra todos los modelos, no solo los compatibles con generateContent.",
    )
    parser.add_argument(
        "--probar",
        metavar="MODELO",
        help="Hace una llamada mínima de prueba con un modelo, por ejemplo gemini-2.5-pro.",
    )
    parser.add_argument(
        "--crear-secrets",
        action="store_true",
        help="Crea .streamlit/secrets.toml con una plantilla local si no existe.",
    )
    args = parser.parse_args()

    try:
        if args.crear_secrets:
            crear_secrets_local()
        elif args.probar:
            probar_modelo(args.probar)
        else:
            listar_modelos(solo_generate_content=not args.todos)
    except RuntimeError as error:
        print(error, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
