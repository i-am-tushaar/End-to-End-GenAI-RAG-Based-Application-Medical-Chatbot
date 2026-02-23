from flask import Flask, render_template, request
from src.rag_pipeline import create_rag_chain

app = Flask(__name__)

# Initialize RAG once
rag_chain = create_rag_chain()


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    user_message = request.form["msg"]

    # ✅ unique memory per user
    session_id = request.remote_addr

    response = rag_chain.invoke(
        {"input": user_message},
        config={"configurable": {"session_id": session_id}}
    )

    return str(response["answer"])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)