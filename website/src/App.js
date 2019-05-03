import React, { Component } from "react";
import logo from "./logo.png";
import "./App.css";
import { Form, Button, TextArea, Icon } from "semantic-ui-react";
import "semantic-ui-css/semantic.min.css";

class App extends Component {
  constructor(props) {
    super(props);
    this.state = {
      inputLyrics: "",
      loadingAPI: false
    };
  }

  updateInputLyrics = () => {
    this.setState({
      inputLyrics: document.getElementById("input-lyrics").value
    });
  };

  render() {
    let body;
    if (this.state.answer) {
      body = <p style={{ lineHeight: 1.2 }}>{this.state.answer}</p>;
    } else {
      body = (
        <Form autoComplete="off">
          <TextArea
            id="input-lyrics"
            placeholder="Lyrics..."
            style={{ minHeight: 200, minWidth: 300 }}
            onChange={this.updateInputLyrics}
          />
          <br />
          <Button
            loading={this.state.loadingAPI}
            onClick={this.submit}
            style={{ marginTop: "10px" }}
            disabled={this.state.inputLyrics.trim().length === 0}
            color="blue"
          >
            Submit
          </Button>
        </Form>
      );
    }

    return (
      <div className="App">
        <header className="App-header">
          <img
            src={logo}
            alt="logo"
            style={{
              marginTop: "10px",
              height: "150px"
            }}
          />
          <h1>Lyrics to Chords</h1>
          {body}
        </header>
        {this.getFooter()}
      </div>
    );
  }

  submit = async () => {
    this.setState({ loadingAPI: true });

    let apiUrl =
      "https://wt-9ea9cc716197cc2580930513c1f0b64d-0.sandbox.auth0-extend.com/Lyrics2ChordsProxy";
    if (
      window.location.hostname === "localhost" &&
      window.location.host !== "localhost:3000"
    ) {
      apiUrl = `http://${window.location.host}/to_chords`;
    }

    let resp;
    try {
      resp = await fetch(apiUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          lyrics: this.state.inputLyrics
        })
      });
    } catch (e) {
      this.setState({ loadingAPI: false });
      return;
    }

    this.setState({ loadingAPI: false });

    if (!resp.ok) {
      return;
    }

    const respJson = await resp.json();

    const answer = this.state.inputLyrics
      .split("\n")
      .map((line, i) => {
        if (line.trim().length === 0) {
          return line;
        }
        if (!Array.isArray(respJson.chords[i])) {
          return line;
        }

        return `${line} (${respJson.chords[i].join(" ")})`;
      })
      .map(line => (
        <span>
          {line}
          <br />
        </span>
      ));

    this.setState({
      answer: answer
    });
  };

  getFooter = () => {
    return (
      <div
        style={{
          position: "fixed",
          left: 0,
          bottom: 0,
          width: "100%",
          backgroundColor: "#e7e7e7",
          color: "black",
          textAlign: "center"
        }}
      >
        Made with <span style={{ fontSize: "large", color: "red" }}>♥</span> by
        Gefen Keinan{" "}
        <a
          href="https://github.com/gefenk9"
          target="_blank"
          rel="noopener noreferrer"
          style={{ color: "black" }}
        >
          <Icon name="github" />
        </a>
      </div>
    );
  };
}

export default App;
