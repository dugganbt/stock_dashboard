# stock_dashboard
A dash project to view historical prices of stocks and examine growth of investments of selected portfolio allocations

## Deploying to Vercel

1. Install the CLI: `npm i -g vercel` and log in with `vercel login`.
2. Add your Tiingo key: `vercel env add TIINGO_API_KEY` (paste the value when prompted).
3. Deploy: `vercel --prod` from the project root. The provided `vercel.json` routes all traffic to `api/index.py`, which serves the Dash app using Python 3.9.
4. Optional local preview: export `TIINGO_API_KEY=...` then run `vercel dev` to test locally.
