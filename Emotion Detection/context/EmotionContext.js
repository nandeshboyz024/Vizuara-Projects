import { createContext, useContext, useEffect, useState } from "react";

const EmotionContext = createContext(null);

const EmotionContextProvider = ({ children }) => {
  const [isTrain, setIsTrain] = useState(false);
  const [myData, setMyData] = useState([]);
  const [size, setSize] = useState({
    joy: 0,
    sadness: 0,
    anger: 0,
    fear: 0,
    love: 0,
    surprise: 0,
  });

  return (
    <EmotionContext.Provider
      value={{
        isTrain,
        setIsTrain,
        myData,
        setMyData,
        size,
        setSize,
      }}
    >
      {children}
    </EmotionContext.Provider>
  );
};

const useEmotionContext = () => {
  // Check if the code is running on the client side
  if (typeof window === "undefined") {
    // Return a default value or handle the case accordingly
    return {};
  }
  return useContext(EmotionContext);
};

export { useEmotionContext, EmotionContextProvider };
