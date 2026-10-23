PhysFormer weights (included in this repository):

  Physformer_UBFC_best.pkl     — default at runtime (UBFC fine-tuned, best checkpoint)
  Physformer_VIPL_fold1.pkl    — VIPL fold-1 pretrained (fine-tune init / fallback)

Additional UBFC epoch checkpoints live under:
  physformer/PhysFormer/Physformer_UBFC_finetune/

Runtime search order is implemented in physformer_engine.resolve_weights_path().
