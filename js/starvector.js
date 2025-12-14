/**
 * ComfyUI-StarVector JavaScript Extension
 * Provides SVG preview capabilities in the node interface
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "starvector.preview",
    
    async setup() {
        // Listen for SVG generation completion messages
        api.addEventListener("starvector.svg.preview", (event) => {
            const { node_id, svg_data } = event.detail;
            const node = app.graph._nodes_by_id[node_id];
            if (node && node.svgPreviewWidget) {
                node.svgPreviewWidget.value = svg_data;
                node.setDirtyCanvas(true, true);
            }
        });
    },
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Add SVG preview functionality to SVGPreview and SVGPreviewFromString nodes
        if (nodeData.name === "SVGPreview" || nodeData.name === "SVGPreviewFromString") {
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function(message) {
                onExecuted?.apply(this, arguments);
                
                // The node outputs an image which ComfyUI will handle
                // But we can add additional info here if needed
            };
        }
        
        // Add info display to StarVectorImage2SVG node
        if (nodeData.name === "StarVectorImage2SVG") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                onNodeCreated?.apply(this, arguments);
                
                // Add a text widget to show generation status
                const statusWidget = this.addWidget(
                    "text",
                    "status",
                    "Ready",
                    () => {},
                    { serialize: false }
                );
                statusWidget.disabled = true;
                
                this.statusWidget = statusWidget;
            };
            
            const onExecutionStart = nodeType.prototype.onExecutionStart;
            nodeType.prototype.onExecutionStart = function() {
                onExecutionStart?.apply(this, arguments);
                if (this.statusWidget) {
                    this.statusWidget.value = "Generating SVG...";
                    this.setDirtyCanvas(true, false);
                }
            };
            
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function(message) {
                onExecuted?.apply(this, arguments);
                if (this.statusWidget) {
                    this.statusWidget.value = "SVG Generated ✓";
                    this.setDirtyCanvas(true, false);
                }
            };
        }
        
        // Add model info to StarVectorModelLoader
        if (nodeData.name === "StarVectorModelLoader") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                onNodeCreated?.apply(this, arguments);
                
                // Add info widget
                const infoWidget = this.addWidget(
                    "text",
                    "info",
                    "Select a model to load",
                    () => {},
                    { serialize: false }
                );
                infoWidget.disabled = true;
                
                this.infoWidget = infoWidget;
                
                // Update info when model selection changes
                const modelWidget = this.widgets?.find(w => w.name === "model_name");
                if (modelWidget) {
                    const originalCallback = modelWidget.callback;
                    modelWidget.callback = (value) => {
                        originalCallback?.call(modelWidget, value);
                        this.updateModelInfo(value);
                    };
                    // Initial update
                    this.updateModelInfo(modelWidget.value);
                }
            };
            
            nodeType.prototype.updateModelInfo = function(modelName) {
                const modelInfo = {
                    "starvector-1b-im2svg": "1B params - Fast, good quality",
                    "starvector-8b-im2svg": "8B params - Best quality, slower"
                };
                if (this.infoWidget) {
                    this.infoWidget.value = modelInfo[modelName] || "Unknown model";
                    this.setDirtyCanvas(true, false);
                }
            };
        }
    },
    
    async nodeCreated(node) {
        // Additional per-instance setup if needed
    },
});
