import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { loadAppConfig } from "./config/apiConfig";
import "./styles.css";

loadAppConfig();

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);





