#!/usr/bin/env node
/**
 * Bridge: forwards Claude Desktop's stdio MCP connection to the
 * remote agentmemory HTTP server via mcp-remote.
 *
 * AGENTMEMORY_URL is injected by the manifest from user_config.tailscale_ip.
 * Optional: AGENTMEMORY_TOKEN or AGENTMEMORY_AUTH_HEADER for Bearer auth.
 */

const { spawn } = require("child_process");

const url = process.env.AGENTMEMORY_URL;
if (!url) {
  process.stderr.write(
    "Error: AGENTMEMORY_URL is not set. Re-install the extension and provide your server IP.\n"
  );
  process.exit(1);
}

const args = ["--yes", "mcp-remote@latest", url, "--allow-http"];

// Bearer auth: prefer full header in env (avoids space-mangling on Windows)
const authHeader =
  process.env.AGENTMEMORY_AUTH_HEADER ||
  (process.env.AGENTMEMORY_TOKEN
    ? `Bearer ${process.env.AGENTMEMORY_TOKEN}`
    : null);

if (authHeader) {
  args.push("--header", `Authorization:${authHeader}`);
}

const child = spawn("npx", args, {
  stdio: "inherit",
  env: { ...process.env },
});

child.on("exit", (code) => process.exit(code ?? 0));
child.on("error", (err) => {
  process.stderr.write(`Failed to start mcp-remote: ${err.message}\n`);
  process.exit(1);
});
