import React, { useEffect, useState } from "react";
import Workspace from "@/models/workspace";
import showToast from "@/utils/toast";
import { useParams } from "react-router-dom";

export default function EovPredictOptions({ workspace }) {
  const { slug } = useParams();
  const activeSlug = slug || (workspace ? workspace.slug : null);

  const [loading, setLoading] = useState(false);
  const [isPredictEnabled, setIsPredictEnabled] = useState(false);
  const [baseUrl, setBaseUrl] = useState("");
  const [authType, setAuthType] = useState("bearer");
  const [authCredentials, setAuthCredentials] = useState("");

  useEffect(() => {
    async function fetchConfig() {
      if (!activeSlug) return;
      const wsInfo = await Workspace.bySlug(activeSlug);
      if (wsInfo) {
        setIsPredictEnabled(wsInfo.is_predict_enabled || false);
      }

      const connector = await Workspace.getConnectors(activeSlug);
      if (connector) {
        setBaseUrl(connector.base_url || "");
        setAuthType(connector.auth_type || "bearer");
        setAuthCredentials(connector.auth_credentials || "");
      }
    }
    fetchConfig();
  }, [activeSlug]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!activeSlug) return;

    try {
      setLoading(true);

      // Update Predict feature switch
      await Workspace.update(activeSlug, { is_predict_enabled: isPredictEnabled });

      // Update connector params
      const res = await Workspace.updateConnectors(activeSlug, {
        base_url: baseUrl,
        auth_type: authType,
        auth_credentials: authCredentials
      });

      if (res && res.success === false) {
        showToast(res.message || "Failed to save options.", "error", { clear: true });
        setLoading(false);
        return;
      }

      showToast("EOV Predict API configuration saved successfully.", "success", { clear: true });
      setLoading(false);
    } catch (e) {
      console.error(e);
      showToast(e.message, "error", { clear: true });
      setLoading(false);
    }
  };

  if (!activeSlug) return null;

  return (
    <div className="flex w-full">
      <div className="flex flex-col w-full px-1 md:pb-6 pb-16">
        <form className="w-full" onSubmit={handleSubmit}>
          <div className="w-full flex flex-col py-2">
            <div className="w-full flex flex-col gap-4">

              <div className="flex flex-col pr-10">
                <div className="flex flex-col gap-y-1 mb-4">
                  <label className="text-white text-sm font-bold flex items-center justify-between">
                    Kích hoạt Tính năng Predict & Analytics
                    <label className="relative inline-flex cursor-pointer items-center">
                      <input
                        type="checkbox"
                        checked={isPredictEnabled}
                        onChange={(e) => setIsPredictEnabled(e.target.checked)}
                        className="peer sr-only"
                      />
                      <div className="peer h-6 w-11 rounded-full bg-slate-600 after:absolute after:left-[2px] after:top-[2px] after:h-5 after:w-5 after:rounded-full after:border after:border-gray-300 after:bg-white after:transition-all after:content-[''] peer-checked:bg-theme-home-button-primary peer-checked:after:translate-x-full peer-checked:after:border-white"></div>
                    </label>
                  </label>
                  <p className="text-xs font-normal text-theme-text-secondary">
                    Khi bật tính năng này, bot AI sẽ gọi API bên dưới để lấy số liệu thực tế thay vì trả lời theo RAG thông thường.
                  </p>
                </div>
              </div>

              {isPredictEnabled && (
                <>
                  <div className="flex flex-col pr-10">
                    <div className="flex flex-col gap-y-1 mb-4">
                      <label className="text-white text-sm font-bold">
                        Base URL (API Của Khách Hàng)
                      </label>
                      <p className="text-xs font-normal text-theme-text-secondary">
                        Ví dụ: https://api.saigonwater.vn
                      </p>
                    </div>
                    <input
                      type="url"
                      name="baseUrl"
                      className="border-none bg-theme-settings-input-bg text-white placeholder:text-theme-settings-input-placeholder text-sm rounded-lg focus:outline-primary-button active:outline-primary-button outline-none block w-full p-2.5"
                      placeholder="http://10.0.0.62:8000"
                      value={baseUrl}
                      required={true}
                      autoComplete="off"
                      onChange={(e) => setBaseUrl(e.target.value)}
                      spellCheck={false}
                    />
                  </div>

                  <div className="flex flex-col pr-10">
                    <div className="flex flex-col gap-y-1 mb-4">
                      <label className="text-white font-bold text-sm flex gap-x-2 items-center">
                        <p className="font-bold text-white">Auth Type</p>
                      </label>
                    </div>
                    <select
                      value={authType}
                      onChange={(e) => setAuthType(e.target.value)}
                      className="border-none bg-theme-settings-input-bg border-gray-500 text-white focus:outline-primary-button active:outline-primary-button outline-none text-sm rounded-lg block w-full p-2.5"
                    >
                      <option value="none">Không yêu cầu bảo mật (None)</option>
                      <option value="bearer">Bearer Token</option>
                      <option value="api-key">API Key (Header)</option>
                    </select>
                  </div>

                  <div className="flex flex-col pr-10">
                    <div className="flex flex-col gap-y-1 mb-4">
                      <label className="text-white font-bold text-sm flex gap-x-2 items-center">
                        <p className="font-bold text-white">
                          Token / Mã API (Credentials)
                        </p>
                      </label>
                      <p className="text-xs font-normal text-theme-text-secondary">
                        Bảo mật: Chuỗi này sẽ được gửi kèm trong mỗi request.
                      </p>
                    </div>
                    <input
                      type="password"
                      name="authCredentials"
                      className="border-none bg-theme-settings-input-bg text-white placeholder:text-theme-settings-input-placeholder text-sm rounded-lg focus:outline-primary-button active:outline-primary-button outline-none block w-full p-2.5"
                      placeholder="Mật khẩu bí mật"
                      value={authCredentials}
                      required={authType !== "none"}
                      autoComplete="off"
                      spellCheck={false}
                      onChange={(e) => setAuthCredentials(e.target.value)}
                    />
                  </div>
                </>
              )}

            </div>

            <div className="flex flex-col py-4 pr-10">
              <button
                type="submit"
                disabled={loading}
                className="mt-2 w-full justify-center border-none px-4 py-2 rounded-lg text-dark-text light:text-white text-sm font-bold items-center flex gap-x-2 bg-theme-home-button-primary hover:bg-theme-home-button-primary-hover disabled:bg-theme-home-button-primary-hover disabled:cursor-not-allowed"
              >
                {loading ? "Lưu thông tin..." : "Lưu Cấu Hình"}
              </button>
            </div>

          </div>
        </form>
      </div>
    </div>
  );
}
