import os
from app.routes import app

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    debug = os.name == "nt"
    print(f"Running on port {port} with debug={debug}")
    app.run(host="0.0.0.0", port=port, debug=debug)
