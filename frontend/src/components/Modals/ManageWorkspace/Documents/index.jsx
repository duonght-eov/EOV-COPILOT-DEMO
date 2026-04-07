import { ArrowsDownUp } from "@phosphor-icons/react";
import { useEffect, useState } from "react";
import Workspace from "../../../../models/workspace";
import System from "../../../../models/system";
import showToast from "../../../../utils/toast";
import Directory from "./Directory";
import WorkspaceDirectory from "./WorkspaceDirectory";

// OpenAI Cost per token
// ref: https://openai.com/pricing#:~:text=%C2%A0/%201K%20tokens-,Embedding%20models,-Build%20advanced%20search

const MODEL_COSTS = {
  "text-embedding-ada-002": 0.0000001, // $0.0001 / 1K tokens
  "text-embedding-3-small": 0.00000002, // $0.00002 / 1K tokens
  "text-embedding-3-large": 0.00000013, // $0.00013 / 1K tokens
};

export default function DocumentSettings({ workspace, systemSettings }) {
  const [highlightWorkspace, setHighlightWorkspace] = useState(false);
  const [availableDocs, setAvailableDocs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [workspaceDocs, setWorkspaceDocs] = useState([]);
  const [selectedItems, setSelectedItems] = useState({});
  const [hasChanges, setHasChanges] = useState(false);
  const [movedItems, setMovedItems] = useState([]);
  const [embeddingsCost, setEmbeddingsCost] = useState(0);
  const [loadingMessage, setLoadingMessage] = useState("");
  
  // OCR Review Jobs queue
  const [globalPendingJobs, setGlobalPendingJobs] = useState([]);
  const [currentReviewJob, setCurrentReviewJob] = useState(null);
  const [excludedImages, setExcludedImages] = useState(new Set());
  const [isSubmittingImages, setIsSubmittingImages] = useState(false);
  const [previewImage, setPreviewImage] = useState(null);

  useEffect(() => {
    if (globalPendingJobs.length > 0 && !currentReviewJob) {
      setCurrentReviewJob(globalPendingJobs[0]);
      setExcludedImages(new Set());
    }
  }, [globalPendingJobs, currentReviewJob]);

  const handleRequireImageSelection = (job_id, images) => {
    setGlobalPendingJobs(prev => [...prev, { job_id, images }]);
  };

  const handleContinueSelection = async () => {
    if (!currentReviewJob) return;
    setIsSubmittingImages(true);
    setLoading(true);
    setLoadingMessage("Đang tiếp tục indexing...");
    
    const excluded_paths = Array.from(excludedImages);
    const selected_paths = currentReviewJob.images
      .filter(img => !excludedImages.has(img.img_path))
      .map(img => img.img_path);
      
    const payload = {
      job_id: currentReviewJob.job_id,
      selected_images: selected_paths,
      excluded_images: excluded_paths,
      workspace: workspace.slug
    };

    const { response, data } = await Workspace.indexSelectedImages(workspace.slug, payload);

    setIsSubmittingImages(false);
    setLoading(false);
    setLoadingMessage("");
    if (!response.ok) {
        showToast(data?.error || "Lỗi khi gọi index hình ảnh", "error");
    } else {
        showToast("Indexing hoàn tất!", "success");
        setCurrentReviewJob(null);
        setGlobalPendingJobs(prev => prev.slice(1));
        await fetchKeys(true);
    }
  };

  async function fetchKeys(refetchWorkspace = false) {
    setLoading(true);
    const localFiles = await System.localFiles();
    const currentWorkspace = refetchWorkspace
      ? await Workspace.bySlug(workspace.slug)
      : workspace;

    const documentsInWorkspace =
      currentWorkspace.documents?.map((doc) => doc.docpath) || [];

    // Documents that are not in the workspace
    const availableDocs = {
      ...localFiles,
      items: localFiles.items.map((folder) => {
        if (folder.items && folder.type === "folder") {
          return {
            ...folder,
            items: folder.items.filter(
              (file) =>
                file.type === "file" &&
                !documentsInWorkspace.includes(`${folder.name}/${file.name}`)
            ),
          };
        } else {
          return folder;
        }
      }),
    };

    // Documents that are already in the workspace
    const workspaceDocs = {
      ...localFiles,
      items: localFiles.items.map((folder) => {
        if (folder.items && folder.type === "folder") {
          return {
            ...folder,
            items: folder.items.filter(
              (file) =>
                file.type === "file" &&
                documentsInWorkspace.includes(`${folder.name}/${file.name}`)
            ),
          };
        } else {
          return folder;
        }
      }),
    };

    setAvailableDocs(availableDocs);
    setWorkspaceDocs(workspaceDocs);
    setLoading(false);
  }

  useEffect(() => {
    fetchKeys(true);
  }, []);

  const updateWorkspace = async (e) => {
    e.preventDefault();
    setLoading(true);
    showToast("Updating workspace...", "info", { autoClose: false });
    setLoadingMessage("This may take a while for large documents");

    const changesToSend = {
      adds: movedItems.map((item) => `${item.folderName}/${item.name}`),
    };

    setSelectedItems({});
    setHasChanges(false);
    setHighlightWorkspace(false);
    await Workspace.modifyEmbeddings(workspace.slug, changesToSend)
      .then((res) => {
        if (res.success === "partial" && res.pending_jobs && res.pending_jobs.length > 0) {
           showToast(`Cần duyệt khối ảnh cho ${res.pending_jobs.length} tài liệu`, "info");
           setGlobalPendingJobs(res.pending_jobs);
        } else if (!!res.message) {
          showToast(`Error: ${res.message}`, "error", { clear: true });
          return;
        } else {
          showToast("Workspace updated successfully.", "success", {
            clear: true,
          });
        }
      })
      .catch((error) => {
        showToast(`Workspace update failed: ${error}`, "error", {
          clear: true,
        });
      });

    setMovedItems([]);
    await fetchKeys(true);
    setLoading(false);
    setLoadingMessage("");
  };

  const moveSelectedItemsToWorkspace = () => {
    setHighlightWorkspace(false);
    setHasChanges(true);

    const newMovedItems = [];

    for (const itemId of Object.keys(selectedItems)) {
      for (const folder of availableDocs.items) {
        const foundItem = folder.items?.find((file) => file.id === itemId);
        if (foundItem) {
          newMovedItems.push({ ...foundItem, folderName: folder.name });
          break;
        }
      }
    }

    let totalTokenCount = 0;
    newMovedItems.forEach((item) => {
      const { cached, token_count_estimate } = item;
      if (!cached) {
        totalTokenCount += token_count_estimate;
      }
    });

    // Do not do cost estimation unless the embedding engine is OpenAi.
    if (systemSettings?.EmbeddingEngine === "openai") {
      const COST_PER_TOKEN =
        MODEL_COSTS[
        systemSettings?.EmbeddingModelPref || "text-embedding-ada-002"
        ];

      const dollarAmount = (totalTokenCount / 1000) * COST_PER_TOKEN;
      setEmbeddingsCost(dollarAmount);
    }

    setMovedItems([...movedItems, ...newMovedItems]);

    let newAvailableDocs = JSON.parse(JSON.stringify(availableDocs));
    let newWorkspaceDocs = JSON.parse(JSON.stringify(workspaceDocs));

    for (const itemId of Object.keys(selectedItems)) {
      let foundItem = null;
      let foundFolderIndex = null;

      newAvailableDocs.items = newAvailableDocs.items.map(
        (folder, folderIndex) => {
          const remainingItems = (folder.items || []).filter((file) => {
            const match = file.id === itemId;
            if (match) {
              foundItem = { ...file };
              foundFolderIndex = folderIndex;
            }
            return !match;
          });

          return {
            ...folder,
            items: remainingItems,
          };
        }
      );

      if (foundItem) {
        newWorkspaceDocs.items[foundFolderIndex].items.push(foundItem);
      }
    }

    setAvailableDocs(newAvailableDocs);
    setWorkspaceDocs(newWorkspaceDocs);
    setSelectedItems({});
  };

  return (
    <div className="flex flex-col gap-4">
      <div className="flex upload-modal -mt-6 z-10 relative">
        <Directory
          files={availableDocs}
          setFiles={setAvailableDocs}
          loading={loading}
          loadingMessage={loadingMessage}
          setLoading={setLoading}
          workspace={workspace}
          fetchKeys={fetchKeys}
          selectedItems={selectedItems}
          setSelectedItems={setSelectedItems}
          updateWorkspace={updateWorkspace}
          highlightWorkspace={highlightWorkspace}
          setHighlightWorkspace={setHighlightWorkspace}
          moveToWorkspace={moveSelectedItemsToWorkspace}
          setLoadingMessage={setLoadingMessage}
          onRequireImageSelection={handleRequireImageSelection}
        />
        <div className="upload-modal-arrow">
          <ArrowsDownUp className="text-white text-base font-bold rotate-90 w-11 h-11" />
        </div>
        <WorkspaceDirectory
          workspace={workspace}
          files={workspaceDocs}
          highlightWorkspace={highlightWorkspace}
          loading={loading}
          loadingMessage={loadingMessage}
          setLoadingMessage={setLoadingMessage}
          setLoading={setLoading}
          fetchKeys={fetchKeys}
          hasChanges={hasChanges}
          saveChanges={updateWorkspace}
          embeddingCosts={embeddingsCost}
          movedItems={movedItems}
        />
      </div>

      {currentReviewJob && (
        <div className="w-full bg-theme-bg-secondary border border-theme-modal-border rounded-xl p-4 shadow-xl z-20 transition-all flex flex-col">
          <div className="flex flex-row items-center justify-between mb-4 bg-black/20 p-3 rounded-xl border border-white/5">
             <div className="flex flex-col">
               <span className="text-white text-base font-bold">
                 Kiểm duyệt ({currentReviewJob.images.length} hình ảnh)
               </span>
               <span className="text-white/70 text-sm mt-1">
                 Có <b className="text-white">{currentReviewJob.images.length}</b> ảnh. Hãy chọn các hình ảnh <b className="text-red-400">rác, không quan trọng</b> để loại bỏ: (<b className="text-red-400">{excludedImages.size}</b> ảnh bị loại trừ)
               </span>
             </div>
             <div className="flex gap-2 relative z-50">
                <button type="button" onClick={() => {
                   setCurrentReviewJob(null);
                   setGlobalPendingJobs(prev => prev.slice(1));
                }} className="px-4 py-2 rounded-lg bg-gray-500/20 text-sm font-semibold text-gray-300 hover:bg-gray-500/40">Huỷ bỏ</button>

                <button type="button" onClick={() => setExcludedImages(new Set(currentReviewJob.images.map(i => i.img_path)))} className="px-4 py-2 rounded-lg bg-red-500/10 text-sm font-semibold text-red-400 hover:bg-red-500/30">Loại bỏ tất cả</button>
                <button type="button" onClick={() => setExcludedImages(new Set())} className="px-4 py-2 rounded-lg bg-blue-500/10 text-sm font-semibold text-blue-400 hover:bg-blue-500/30">Giữ tất cả</button>
                <button type="button" onClick={handleContinueSelection} disabled={isSubmittingImages} className="px-6 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-sm font-semibold transition-colors disabled:opacity-50 ml-2">
                  {isSubmittingImages ? "Đang xử lý..." : "Tiếp tục Indexing"}
                </button>
             </div>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-3 overflow-y-auto max-h-[60vh] w-full no-scroll p-1 relative z-50">
             {currentReviewJob.images.map(img => (
                <div key={img.img_path} className="relative cursor-pointer group bg-black/40 rounded-xl overflow-hidden aspect-square flex flex-col items-center justify-center p-1 border border-transparent hover:border-white/20 transition-all shadow-md" onClick={(e) => {
                  e.stopPropagation();
                  const newSet = new Set(excludedImages);
                  if (newSet.has(img.img_path)) newSet.delete(img.img_path);
                  else newSet.add(img.img_path);
                  setExcludedImages(newSet);
                }}>
                  {img.preview_url ? (
                    <img src={img.preview_url} alt="preview" className={`w-full h-full object-cover rounded-lg shadow-sm transition-all duration-300 ${excludedImages.has(img.img_path) ? 'opacity-30 grayscale sepia brightness-50' : 'hover:scale-105'}`} onError={(e) => { e.target.onerror = null; e.target.src = 'data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI0MiIgaGVpZ2h0PSI0MiIgZmlsbD0iI2FhYSIgY2xhc3M9ImJpIGJpLWV4Y2xhbWF0aW9uLXRyaWFuZ2xlIiB2aWV3Qm94PSIwIDAgMTYgMTYiPjxwYXRoIGQ9Ik03LjkzOCIuMDIzQzczLjAxNnoiLz48L3N2Zz4='; }} />
                  ) : (
                    <span className="text-xs text-white/50">No Image</span>
                  )}
                  {excludedImages.has(img.img_path) && (
                    <div className="absolute mt-0 flex items-center justify-center pointer-events-none">
                      <div className="bg-red-500/90 text-white rounded-full w-12 h-12 flex items-center justify-center text-3xl font-bold shadow-2xl ring-4 ring-red-500/30">✗</div>
                    </div>
                  )}
                  <div className="absolute top-2 right-2 bg-black/80 text-[10px] text-white px-2 py-1 rounded-md shadow-lg pointer-events-none z-10 border border-white/10">Trg {img.page_number}</div>
                  <div className="absolute bottom-2 left-2 right-2 flex justify-center opacity-0 group-hover:opacity-100 transition-opacity z-20">
                     <button className="bg-black/80 text-white text-xs px-3 py-1.5 rounded-lg border border-white/20 hover:bg-black font-semibold" onClick={(e) => { e.stopPropagation(); setPreviewImage(img.preview_url); }}>Phóng to</button>
                  </div>
                </div>
             ))}
          </div>
        </div>
      )}

      {/* Modal Zoom Hình To */}
      {previewImage && (
        <div className="fixed inset-0 z-[9999] bg-black/90 flex items-center justify-center backdrop-blur-sm" onClick={() => setPreviewImage(null)}>
           <span className="absolute top-4 right-6 text-white text-5xl cursor-pointer hover:text-red-400 drop-shadow-md">&times;</span>
           <img src={previewImage} className="max-w-[90vw] max-h-[90vh] object-contain rounded-xl shadow-2xl border border-white/10" onClick={(e) => e.stopPropagation()} />
        </div>
      )}
    </div>
  );
}
