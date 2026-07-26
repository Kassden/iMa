# Use a static modular run-analysis dashboard

Date: 2026-07-27

## Status
proposed

## Context
The current dashboard embeds a large evolving JSON payload and all CSS, state, rendering, custom canvas charts, and Plotly charts in one generated HTML file. Latest-only notebook data and schema-level feature studies bypass the selected-run state, producing contradictory scopes. The deployment is a read-only static Vercel site and does not require server rendering or an application backend.

## Decision
Retain static Vercel deployment, introduce a Python dashboard-contract module that publishes a normalized analysis view model, split the browser into ES modules for state, selectors, views, and Plotly charts, fetch the versioned results JSON at runtime, and use Plotly.js as the only charting framework. Keep training, scraping, and model policy outside the presentation layer.

## Consequences
The redesign avoids a framework migration and preserves simple deployment while making data scope testable and UI state coherent. It adds a versioned publication contract and several small static assets. Browser tests must verify contract compatibility, URL state, trace data, and responsive behavior. Plotly remains a pinned client dependency.
