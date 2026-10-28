/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          950: "#08111f",
          900: "#0d1929",
          800: "#13243a",
          700: "#1b3552",
        },
        brand: {
          50: "#eef8ff",
          100: "#d9efff",
          200: "#bce4ff",
          300: "#8ed4ff",
          400: "#59bbff",
          500: "#329dff",
          600: "#197fe8",
          700: "#1566c3",
          800: "#18579d",
          900: "#194b7d",
        },
      },
      fontFamily: {
        sans: ["Inter", "PingFang SC", "Microsoft YaHei", "system-ui", "sans-serif"],
      },
      boxShadow: {
        panel: "0 18px 60px rgba(15, 35, 62, 0.10)",
      },
    },
  },
  plugins: [],
};
