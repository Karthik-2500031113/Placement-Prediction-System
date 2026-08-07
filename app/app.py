from flask import Flask, render_template
from src.data.load_data import load_data

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/dataset")
def dataset():
    df = load_data()

    return render_template(
        "dataset.html",
        rows=df.shape[0],
        cols=df.shape[1],
        columns=df.columns.tolist(),
        data=df.head().values.tolist()
    )


@app.route("/eda")
def eda():
    return render_template("eda.html")


if __name__ == "__main__":
    app.run(debug=True)