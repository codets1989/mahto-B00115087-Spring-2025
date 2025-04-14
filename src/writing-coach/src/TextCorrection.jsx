import React, { useState } from "react";
import './index.css';

function TextCorrection() {
  const [inputText, setInputText] = useState("");
  const [correctedText, setCorrectedText] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setCorrectedText("");

    try {
      const response = await fetch("http://localhost:5278/api/grammar/correct", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: inputText }),
      });

      const data = await response.json();
      console.log(data);
      setCorrectedText(data.corrected_text);
    } catch (err) {
      console.error(err);
      setCorrectedText("Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-100 to-purple-100 flex items-center justify-center p-4">
      <div className="bg-white shadow-xl rounded-2xl p-8 max-w-xl w-full space-y-6">
        <h1 className="text-2xl font-bold text-gray-800 text-center">✨ Text Correction Tool</h1>

        <form onSubmit={handleSubmit} className="space-y-4">
          <textarea
            className="w-full p-4 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-400 transition"
            rows="4"
            placeholder="Type something like: 'Teh cat is on teh mat.'"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
          />

          <button
            type="submit"
            disabled={loading || inputText.trim() === ""}
            className="w-full bg-purple-600 hover:bg-purple-700 text-black font-semibold py-3 rounded-lg shadow-md transition duration-300 disabled:opacity-50"
          >
            {loading ? "Correcting..." : "Correct Text"}
          </button>
        </form>

        {correctedText && (
          <div className="bg-green-50 border-l-4 border-green-400 text-green-800 p-4 rounded-md shadow-sm">
            <strong>Corrected:</strong> <span className="italic">{correctedText}</span>
          </div>
        )}
      </div>
    </div>
  );
}

export default TextCorrection;
