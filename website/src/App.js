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
    let chords;
    if (this.state.chords) {
      chords = <pre>{JSON.stringify(chords, null, 4)}</pre>;
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
        </header>
        {chords}
        {this.getFooter()}
      </div>
    );
  }

  submit = async () => {
    this.setState({ loadingAPI: true });

    let apiUrl =
      "http://ec2-34-245-176-135.eu-west-1.compute.amazonaws.com/to_chords";
    // if (window.location.hostname === "localhost") {
    //   apiUrl = `http://${window.location.host}/to_chords`;
    // }

    const resp = await fetch(apiUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        lyrics: this.state.inputLyrics
      })
    });

    if (!resp.ok) {
      this.setState({ loadingAPI: false });
      return;
    }

    const respJson = await resp.json();

    this.setState({
      chords: respJson.chords
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
