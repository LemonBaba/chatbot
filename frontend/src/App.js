import React, { useState } from 'react';
import './App.css';

function App() {
  const [userInput, setUserInput] = useState('');
  const [chatHistory, setChatHistory] = useState([]);

  const handleSubmit = async (e) => {
    e.preventDefault();

    // Send user input to backend
    const response = await fetch('http://localhost:5000/chatbot', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ user_input: userInput }),
    });

    const data = await response.json();

    setChatHistory([
      ...chatHistory,
      { text: userInput, user: true },
      { text: data.response, user: false },
    ]);

    // Reset input field
    setUserInput('');
  };

  const handleChange = (e) => {
    setUserInput(e.target.value);
  };

  return (
    <div className="App">
      <h1>Chatbot</h1>
      <div className="chat-container">
        {chatHistory.map((message, index) => (
          <div key={index} className={message.user ? 'user-message' : 'bot-message'}>
            {message.text}
          </div>
        ))}
        <form className="form" onSubmit={handleSubmit}>
          <input
            type="text"
            value={userInput}
            onChange={handleChange}
            placeholder="Type your message..."
          />
          <button type="submit">Send</button>
        </form>
      </div>
    </div>
  );
}

export default App;
