# Vektora Auto-Trade Bot

Auto-trades Binance Futures based on [Vektora](https://vektora.trade) signals, 24/7.

## Deploy to Railway

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.com/deploy/vektora-bot-template?referralCode=rWoFxF)

## Setup

1. Subscribe at [vektora.trade](https://vektora.trade) to get your Signal API key
2. Create a Binance Futures API key — whitelist IP `208.77.244.15`
3. Click the deploy button above
4. Set these 3 variables in Railway:

| Variable | Description |
|----------|-------------|
| `SIGNAL_API_KEY` | Your Vektora API key (from subscription email) |
| `BINANCE_KEY` | Binance Futures API key |
| `BINANCE_SECRET` | Binance Futures API secret |

5. Deploy — the bot auto-configures and starts trading

## Optional: Telegram Alerts

Add these variables to get trade notifications on Telegram:

| Variable | Description |
|----------|-------------|
| `TELEGRAM_BOT_TOKEN` | Create a bot via [@BotFather](https://t.me/BotFather) |
| `TELEGRAM_CHAT_ID` | Get your ID from [@userinfobot](https://t.me/userinfobot) |

## How It Works

- Connects to Vektora signal server via WebSocket
- Receives direction-flip signals for 18 crypto futures pairs
- Executes trades on your Binance account via secure proxy
- 5% risk per trade, 10x leverage, 8% stop-loss on every position
- Runs 24/7 on Railway — auto-reconnects if disconnected

## Security

- Your Binance API keys stay on YOUR Railway instance
- Keys are never stored on Vektora servers
- Trades execute through an isolated proxy with IP whitelisting
- The proxy signs requests in memory — never stores credentials

## API Endpoints

- `GET /health` — Bot status (no auth required)

## Full Setup Guide

[vektora.trade/guide](https://vektora.trade/guide) — step-by-step with Binance account setup, API key creation, and troubleshooting.

## Support

- Email: support@vektora.trade
- Telegram: [@donchian_alert_bot](https://t.me/donchian_alert_bot)
