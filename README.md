# AskMyDocs — Document Duel

TwoDocsChallenge submission: upload two PDFs, both get indexed into Azure AI Search,
and each question is answered by Azure OpenAI using whichever document's chunks
matched best. The UI frames it as a "duel" — the winning document lights up.

## Deploy via Azure Portal + GitHub Actions (using keys you already have)

You don't need to create your own Azure AI Search or OpenAI resource — just plug
the given key/endpoint into an App Service.

1. **Create the App Service** (portal.azure.com → App Services → Create):
   - Publish: Code · Runtime stack: Python 3.10 · OS: Linux
   - Pick your resource group + region, then Review + create.

2. **Add environment variables**: App Service → Settings → Environment variables → Add:
   - `AZURE_SEARCH_ENDPOINT`, `AZURE_SEARCH_KEY`, `AZURE_SEARCH_INDEX`
   - `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_KEY`, `AZURE_OPENAI_DEPLOYMENT`, `AZURE_OPENAI_API_VERSION`
   - Save, then confirm/restart when prompted.

3. **Enable Basic Auth + get the publish profile**: Settings → Configuration →
   General settings → turn on Basic Auth Publishing Credentials → Save. Then go to
   Overview → **Download publish profile**.

4. **Set the startup command**: Configuration → Stack settings → Startup Command:
   ```
   gunicorn --bind 0.0.0.0:8000 --timeout 180 main:app
   ```

5. **Add the publish profile as a GitHub secret**: your repo → Settings → Secrets
   and variables → Actions → New repository secret:
   - Name: `AZURE_WEBAPP_PUBLISH_PROFILE`
   - Value: the full contents of the downloaded publish profile file

6. **Update the app name** in `.github/workflows/deploy.yml` (`app-name:`) to match
   your App Service's name exactly.

7. **Test locally first** (`pip install -r requirements.txt`, set the same env vars
   in a local `.env`, `python main.py`), then commit everything — code + the
   `.github/workflows/deploy.yml` file — in one push to `main`.

8. GitHub Actions triggers automatically and deploys. Your app is live at
   `https://<app-name>.azurewebsites.net`.

## Local `.env` template

```
AZURE_SEARCH_ENDPOINT=https://<your-search>.search.windows.net
AZURE_SEARCH_KEY=<key>
AZURE_SEARCH_INDEX=documents
AZURE_OPENAI_ENDPOINT=https://<your-openai>.openai.azure.com/
AZURE_OPENAI_KEY=<key>
AZURE_OPENAI_DEPLOYMENT=<deployment-name>
AZURE_OPENAI_API_VERSION=2024-12-01-preview
```

## How the duel is decided

Each question runs a search over both documents' chunks. The top 3 matches are
sent to Azure OpenAI as grounding context. Whichever document contributed more of
those top 3 chunks is marked the "winner" and its color glows in the UI.