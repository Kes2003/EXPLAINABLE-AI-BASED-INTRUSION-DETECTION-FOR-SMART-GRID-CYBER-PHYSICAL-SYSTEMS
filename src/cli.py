from __future__ import annotations

import argparse
import uvicorn


def main():
	parser = argparse.ArgumentParser(description="Smart Grid IDS CLI")
	parser.add_argument("command", choices=["serve"], help="Action to run")
	parser.add_argument("--host", default="0.0.0.0")
	parser.add_argument("--port", type=int, default=8000)
	args = parser.parse_args()

	if args.command == "serve":
		uvicorn.run("src.api:app", host=args.host, port=args.port, reload=False, workers=1)


if __name__ == "__main__":
	main()






