import React from 'react';
import { BrowserRouter as Router } from 'react-router-dom';
import SideBar from './Presentation/Common/Sidebar'; 
import AppRouter from './Routes/AppRouter'; 
import './style/App.css'; 
function App() {
  return (
    <Router>
      <div className="app-container">
        <SideBar />
        <div className="content-container">
          <AppRouter /> 
        </div>
      </div>
    </Router>
  );
}

export default App;