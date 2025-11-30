# EXAID Next.js Frontend

Modern Next.js 14 (App Router) implementation of the EXAID Clinical Decision Support System frontend with real-time WebSocket token streaming and intelligent summarization.

## Features

- ✅ **Real-time Token Streaming**: Live agent reasoning traces with 50ms batching optimization
- ✅ **Smart Auto-Scroll**: Maintains scroll position with 50px threshold detection
- ✅ **Summary Accordion**: Only one expanded summary at a time for cognitive load management
- ✅ **WebSocket Resilience**: Automatic reconnection with fixed 3s delay (max 5 attempts)
- ✅ **Accessible Modal**: Focus lock, ESC key, backdrop click, ARIA labels
- ✅ **Responsive Design**: Mobile-first with breakpoint at 768px
- ✅ **Performance Optimized**: React.memo, Zustand selectors, token batching

## Tech Stack

- **Framework**: Next.js 14 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **State Management**: Zustand
- **Animations**: Framer Motion
- **Accessibility**: react-focus-lock

## Quick Start

### Prerequisites

- Node.js 18+
- FastAPI backend running on `localhost:8000`

### Installation

```bash
# Install dependencies (already done)
npm install

# Run development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Environment Variables

Create a `.env.local` file in the `exaid-frontend` directory with the following variables:

```env
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws
NEXT_PUBLIC_API_URL=http://localhost:8000
```

**Note:** The `.env.local` file is excluded from version control (via `.gitignore`) to protect sensitive configuration. Each developer must create their own local copy.

**WebSocket URL Validation:** The application validates that `NEXT_PUBLIC_WS_URL` starts with `ws://` or `wss://` on startup. An invalid URL will cause a clear error message.

## Project Structure

```
├── app/
│   ├── layout.tsx              # Root layout with modal portal
│   ├── page.tsx                # Main page with WebSocket init
│   └── globals.css             # Global Tailwind styles
├── components/
│   ├── Header.tsx              # Connection status header
│   ├── CaseInput.tsx           # Auto-resize textarea form
│   ├── AgentWindow.tsx         # Memoized agent trace window
│   ├── AgentWindowContent.tsx  # Auto-scroll implementation
│   ├── AgentTracesPanel.tsx    # Left panel container
│   ├── SummaryCard.tsx         # Memoized summary card
│   ├── SummariesPanel.tsx      # Right panel container
│   └── AgentModal.tsx          # Portal-based modal
├── lib/
│   ├── types.ts                # TypeScript interfaces
│   └── websocket.ts            # Singleton WebSocket service with URL validation
├── store/
│   └── cdssStore.ts            # Zustand global state
├── next.config.mjs             # Next.js config
└── tailwind.config.ts          # Tailwind theme
```

## Running Both Servers

You need TWO terminal windows:

**Terminal 1 - FastAPI Backend:**
```powershell
cd c:\Users\abemk\source\repos\AbemKW\ExAID
python -m uvicorn web_ui.server:app --reload
```

**Terminal 2 - Next.js Frontend:**
```powershell
cd c:\Users\abemk\source\repos\AbemKW\ExAID\exaid-frontend
npm run dev
```

## Key Implementation Details

### WebSocket Service (`lib/websocket.ts`)
- Singleton pattern prevents duplicate connections
- Fixed 3000ms reconnection delay (NOT exponential backoff)
- Max 5 reconnection attempts
- Connection state safeguards

### State Management (`store/cdssStore.ts`)
- Zustand store with `subscribeWithSelector` middleware
- Token buffering with 50ms flush interval
- Selector hooks prevent cross-agent re-renders
- Global reset on `processing_started` message

### Auto-Scroll (`components/AgentWindowContent.tsx`)
- 50px threshold for "at bottom" detection
- `requestAnimationFrame` for smooth scrolling
- Respects user manual scrolling

## Development Commands

```bash
npm run dev      # Development server
npm run build    # Production build
npm run start    # Start production server
npm run lint     # Lint check
```

## Troubleshooting

### WebSocket Won't Connect
- Verify FastAPI running on port 8000
- Check CORS middleware added to backend (see MIGRATION_GUIDE.md)
- Confirm `.env.local` has correct URLs

### Duplicate Connections in Dev Tools
- `reactStrictMode: false` in `next.config.ts` is intentional
- Dev-only behavior; production won't have this issue

## Documentation

See `../MIGRATION_GUIDE.md` for comprehensive migration documentation, verification checklist, and deployment instructions.

## License

Same as parent EXAID project.
