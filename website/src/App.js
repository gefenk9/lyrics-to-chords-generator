import React, { Component } from "react";
import logo from "./logo.png";
import "./App.css";
import { Form, Button, TextArea, Icon } from "semantic-ui-react";
import "semantic-ui-css/semantic.min.css";

class App extends Component {
  constructor(props) {
    super(props);
    this.state = {
      isLoading: false,
      mainImage: logo
    };
  }

  getFooter() {
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
          style={{ color: "black" }}
        >
          <Icon name="github" />
        </a>
      </div>
    );
  }

  render() {
    let body = (
      <div className="App">
        <header className="App-header">
          <img
            src={this.state.mainImage}
            alt="logo"
            style={{
              marginTop: "10px",
              height: "150px"
            }}
          />
          <h1>Lyrics to Chords</h1>
          <Form autoComplete="off">
            <TextArea
              placeholder="Lyrics..."
              style={{ minHeight: 200, minWidth: 300 }}
            />
            <br />
            <Button
              loading={this.state.loadingAPI}
              onClick={this.submit}
              style={{ marginTop: "10px" }}
              disabled={
                !this.state.uploadedImageB64 ||
                !this.state.username ||
                this.state.usernameError
              }
              color="blue"
            >
              Submit
            </Button>
          </Form>
        </header>
        {this.getFooter()}
      </div>
    );
    return body;
  }
}

export default App;
