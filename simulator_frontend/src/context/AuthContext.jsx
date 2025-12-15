import React, { createContext, useState, useContext, useEffect } from 'react';

const AuthContext = createContext(null);

// Helper function to decode JWT and check expiration
const isTokenExpired = (token) => {
    if (!token) return true;

    try {
        const payload = JSON.parse(atob(token.split('.')[1]));
        const expirationTime = payload.exp * 1000; // Convert to milliseconds
        return Date.now() >= expirationTime;
    } catch (error) {
        console.error('Error decoding token:', error);
        return true;
    }
};

export const AuthProvider = ({ children }) => {
    const [user, setUser] = useState(null);
    const [token, setToken] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        // Check for stored token on mount
        const storedToken = localStorage.getItem('token');
        const storedUser = localStorage.getItem('user');

        if (storedToken && storedUser) {
            // Check if token is expired
            if (isTokenExpired(storedToken)) {
                console.log('Token expired, clearing session');
                localStorage.removeItem('token');
                localStorage.removeItem('user');
                setToken(null);
                setUser(null);
            } else {
                setToken(storedToken);
                setUser(JSON.parse(storedUser));
            }
        }
        setLoading(false);
    }, []);

    const login = (token, userData) => {
        setToken(token);
        setUser(userData);
        localStorage.setItem('token', token);
        localStorage.setItem('user', JSON.stringify(userData));
    };

    const logout = () => {
        setToken(null);
        setUser(null);
        localStorage.removeItem('token');
        localStorage.removeItem('user');
    };

    // Check token expiration periodically (every 30 seconds)
    useEffect(() => {
        if (!token) return;

        const checkTokenInterval = setInterval(() => {
            if (isTokenExpired(token)) {
                console.log('Token expired during session, logging out');
                logout();
                alert('Your session has expired. Please log in again.');
            }
        }, 30000); // Check every 30 seconds

        return () => clearInterval(checkTokenInterval);
    }, [token]);

    const value = {
        user,
        token,
        loading,
        login,
        logout,
        isAuthenticated: !!token,
        isTokenExpired // Export function for manual checks
    };

    return (
        <AuthContext.Provider value={value}>
            {children}
        </AuthContext.Provider>
    );
};

export const useAuth = () => {
    const context = useContext(AuthContext);
    if (!context) {
        throw new Error('useAuth must be used within an AuthProvider');
    }
    return context;
};