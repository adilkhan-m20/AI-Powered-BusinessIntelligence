import { useState } from "react";
import { useAuthStore } from "../store/useAuthStore";
import { Link } from "react-router-dom";
import { Eye, EyeOff, Loader2, Lock, Mail, User } from "lucide-react";
import toast from "react-hot-toast";
import signUp from "../assets/signUp.jpg";

const SignUpPage = () => {
  const [showPassword, setShowPassword] = useState(false);
  const [formData, setFormData] = useState({
    fullname: "",
    email: "",
    password: "",
  });
  const { signup, isSigningUp } = useAuthStore();

  const validateForm = () => {
    if (!formData.fullName.trim()) return toast.error("Full name is required");
    if (!formData.email.trim()) return toast.error("Email is required");
    if (!/\S+@\S+\.\S+/.test(formData.email))
      return toast.error("Invalid email format");
    if (!formData.password) return toast.error("Password is required");
    if (formData.password.length < 6)
      return toast.error("Password must be at least 6 characters");

    return true;
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    const success = validateForm();
    if (success === true) signup(formData);
  };

  return (
    <div className="h-screen grid lg:grid-cols-2">
      {/**Left side of the form */}
      <div className="flex flex-col items-start pl-12">
        <h1 className="text-2xl font-bold pt-16">
          Welcome to Business Intelligence
        </h1>
        <p className="text-base-content/80 text-xs">
          Already have an account ?{" "}
          <Link to="/login" className="link link-primary">
            Log in
          </Link>
        </p>

        <form onSubmit={handleSubmit} className="space-y-10 pt-10">
          <div className="form-control">
            <label className="label flex flex-row space-x-2">
              <span className="label-text font-medium">Full Name</span>
              <User className="size-5 text-base-content/40" />
            </label>
            <div className="relative pt-2">
              <input
                type="text"
                className={"input input-bordered w-96"}
                placeholder="Adil Khan"
                value={formData.fullName}
                onChange={(e) =>
                  setFormData({ ...formData, fullname: e.target.value })
                }
              />
            </div>
          </div>

          <div className="form-control">
            <label className="label flex flex-row space-x-2">
              <span className="label-text font-medium">Email</span>
              <Mail className="size-5 text-base-content/40" />
            </label>
            <div className="relative pt-2">
              <input
                type="email"
                className={"input input-bordered w-96 "}
                placeholder="adilkhan@example.com"
                value={formData.email}
                onChange={(e) =>
                  setFormData({ ...formData, email: e.target.value })
                }
              />
            </div>
          </div>

          <div className="form-control">
            <div className="label flex flex-row space-x-2">
              <span className="label-text font-medium">Password</span>
              <Lock className="size-5 text-base-content/40" />
              <button
                type="button"
                className="pl-60"
                onClick={() => setShowPassword(!showPassword)}
              >
                {showPassword ? (
                  <EyeOff className="size-5 text-base-content/40" />
                ) : (
                  <Eye className="size-5 text-base-content/40" />
                )}
              </button>
            </div>
            <div className="relative pt-2">
              <input
                type={showPassword ? "text" : "password"}
                className={"input input-bordered w-96"}
                placeholder="••••••••"
                value={formData.password}
                onChange={(e) =>
                  setFormData({ ...formData, password: e.target.value })
                }
              />
              <p className="text-base-content/65 font-extralight text-sm pt-2">
                • Password must be at least 6 characters
              </p>
            </div>
          </div>

          <button
            type="submit"
            className="btn btn-primary w-36 rounded-3xl"
            disabled={isSigningUp}
          >
            {isSigningUp ? (
              <>
                <Loader2 className="size-5 animate-spin" />
                Loading ...
              </>
            ) : (
              "Create an Account"
            )}
          </button>
          <p className=" -mt-96 text-pretty text-xs text-base-content/75">
            Already have an Account ?{" "}
            <Link to="/login" className="link link-primary">
              Log in
            </Link>
          </p>
        </form>
      </div>
      {/**Right side of the form */}
      <div className="relative w-full h-fit overflow-hidden">
        <img
          src={signUp}
          alt="Sign Up Image"
          className="w-full h-auto block object-cover object-top"
        />
        <div className="absolute left-4 bottom-4 text-left pb-32">
          <div className="p-4">
            <h2 className="text-white text-2xl font-bold drop-shadow-md">
              Join now to Increase you Intelligence
            </h2>
            <p className="text-white/90 drop-shadow">you Mother Fucker</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SignUpPage;
