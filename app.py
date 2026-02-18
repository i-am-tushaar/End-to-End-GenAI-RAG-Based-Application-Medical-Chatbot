# app.py

from flask import Flask, render_template, request
from src.rag_pipeline import create_rag_chain

app = Flask(__name__)

# initialize RAG once at startup
rag_chain = create_rag_chain()


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    user_message = request.form["msg"]

    response = rag_chain.invoke({
        "input": user_message
    })

    return str(response["answer"])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
