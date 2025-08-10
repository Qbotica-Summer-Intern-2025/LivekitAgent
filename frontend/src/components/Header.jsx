import React from "react";

const Header = () => {
  return (
    <div className="fixed top-0 left-0 right-0 flex items-center justify-between px-4 py-2 bg-white shadow-md w-full h-14 z-50">
      {/* Left: Logo and Text */}
      <div className="flex items-center space-x-2">
        <img src="/img_1.png" alt="qBotica Logo" className="h-6 w-6" />
        <span className="font-semibold text-lg">qBotica</span>
      </div>

      {/* Center: title */}
      <h1 className="font-bold text-gray-500" style={{ fontSize: '30px' }}>Polaris</h1>





      {/* Right: End Session Button */}
      <button className="bg-orange-500 text-white px-4 py-1 rounded hover:bg-orange-600 transition-colors">
        End Session
      </button>
    </div>
  );
};

export default Header;