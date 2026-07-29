from flask import Flask, render_template

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("home.html")


@app.route("/dataset")
def dataset():
    return render_template("dataset.html")


@app.route("/eda")
def eda():
    return render_template("eda.html")


@app.route("/preprocessing")
def preprocessing():
    return render_template("preprocessing.html")


@app.route("/models")
def models():
    return render_template("models.html")


@app.route("/results")
def results():
    return render_template("results.html")


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )