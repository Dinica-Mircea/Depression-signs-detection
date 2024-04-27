import React, { useState, useEffect } from 'react';
import './index.css';

const App: React.FC = () => {
    const [textInput, setTextInput] = useState('');
    const [result, setResult] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState<boolean>(false);
    const [displayedResult, setDisplayedResult] = useState('');

    useEffect(() => {
        if (result) {
            setDisplayedResult('');

            // Starting the typing effect
            const timer = setInterval(() => {
                setDisplayedResult((prev) => {
                    if (prev.length < result.length) {
                        return prev + result.charAt(prev.length);
                    } else {
                        clearInterval(timer); // Stop the timer if the end of the string is reached
                        return prev;
                    }
                });
            }, 50); // Typing speed

            // Clean up the interval on unmount
            return () => clearInterval(timer);
        }
        // Make sure to clear the displayedResult when 'result' becomes null or changes
        else {
            setDisplayedResult('');
        }
    }, [result]);



    const handleCheck = async () => {
        console.log("started handle check")
        setResult('')
        if (!textInput.trim()) {
            setResult('Please enter some text to check.');
            return;
        }
        setIsLoading(true); // Set loading to true when request starts

        try {
            const response = await fetch('http://127.0.0.1:5000/classify', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: textInput })
            });
            const data = await response.json();
            if (data.result === "depression") {
                setResult("Signs of depression detected. It might be helpful to reach out and talk more with the person who wrote this message. Offering a listening ear can make a big difference.");
            } else {
                setResult("It appears there are no signs of depression, which is reassuring. Thank you for taking the time to look out for the well-being of the person who wrote this message.");
            }
        } catch (error) {
            console.error('Error:', error);
            setResult('Error processing your request.');
        }
        setIsLoading(false); // Set loading to false when request completes
    };

    return (
        <div className="app-container">
            <h1>Depression Signs Detector</h1>
            <textarea className="text-input" value={textInput} onChange={(e) => setTextInput(e.target.value)} placeholder="Enter text here..." />
            <button className="check-button" onClick={handleCheck} disabled={isLoading}>Check for Depression Signs</button>
            {isLoading ? <div className="loader"></div> : null}
            <div className="result">{displayedResult}</div>
        </div>
    );
};

export default App;
