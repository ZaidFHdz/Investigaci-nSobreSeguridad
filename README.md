# Investigación: Seguridad en México

Dashboard en Streamlit para explorar percepción de inseguridad, cifra negra e incidencia delictiva en México.

## Ejecutar localmente

```bash
pip install -r requirements.txt
streamlit run app_dashboard.py
```

## Secretos

La app usa Gemini solo si existe `GEMINI_API_KEY` en secretos de Streamlit.

En local, crea `.streamlit/secrets.toml` sin subirlo a GitHub:

```toml
GEMINI_API_KEY = "tu_api_key"
```
