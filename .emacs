(add-to-list 'load-path "~/.emacs.d/lisp/")
(load "/usr/local/share/gjdb/emacs/lisp/gjdb")

(require 'cc-mode)
(require 'compile)
(require 'newcomment)
(require 'fill-column-indicator)

(setq comment-auto-fill-only-comments 1)
(setq-default auto-fill-function 'do-auto-fill)

(setq-default indent-tabs-mode nil)
(set-default-font "Menlo 14")
(global-linum-mode 1)
(setq column-number-mode t)
(load-theme 'tango-dark)

(global-set-key (kbd "M-S-<left>") 'previous-buffer)
(global-set-key (kbd "M-S-<right>") 'next-buffer)

(global-set-key (kbd "C-x O") 'previous-multiframe-window)
(global-set-key (kbd "C-x a") 'previous-multiframe-window)
(global-set-key (kbd "<M-down>") 'forward-paragraph)
(global-set-key (kbd "<M-up>")   'backward-paragraph)

;moves saves to .saves folder
(setq backup-directory-alist `(("." . "~/.saves")))

;turns on emacs fill column indicator, but only for code
(define-globalized-minor-mode global-fci-mode fci-mode
  (lambda ()
    (if (and
         (not (string-match "^\*.*\*$" (buffer-name)))
         (not (eq major-mode 'dired-mode)))
        (fci-mode 1))))
(global-fci-mode 1)
(setq fci-rule-column 80)



;Tries to use the local mark, if it doesn't have one, then use global                                        
;(defun pop-local-or-global-mark ()
;  "Pop to local mark if it exists or to the global mark if it does not."
;  (interactive)
;  (if (mark t)
;      (pop-to-mark-command)
;      (pop-global-mark)))

