;;Простая, но полноценная библиотека для построения нейронных сетьей с обучением, входными, скрытыми и выходными слоями,
;;алгоритмом обратного распространения, нейронами смещения. Идея реализации в виде связной сети,
;;а не операций с матрицами, взята из SICP(структура и интерпретация компьютерных программ).
;;Вся структура данных и функций построена на замыканиях.



;;Списки нейронов, синапсов
(defvar *neurons* '())
(defvar *syns* '())
(defvar *in-syns* '())
(defvar *out-syns* '())

;;коэффициенты для метода град. спуска
(defvar *eps* 0.7) ;скорость обучения
(defvar *alpha* 0.3) ;момент


(defvar *in-nrns* '())
(defvar *hid-nrns* '())
(defvar *out-nrns* '())


(defvar *bias-in-syns* '())
(defvar *transf* nil)
(defvar *transf-diff* nil)




(defmacro prc (obj f &rest args)
  (if args
    `(funcall (funcall ,obj ,f) ,@args)
    `(funcall ,obj ,f)))

(defun prcl (objects args)
  (loop for x in objects collect
    (prc x args)))

;;Тело нейрона(сома) - основная структура, замыкания во все края
(defun neuron (&optional in-or-out-neuron? name)
  (let ((num-in-syns 0)
        (num-out-syns 0)
        (in-val 0)
        (out-val 0)
        (delta 0)
        (num-syns-has-value 0)
        (num-syns-has-delta-value 0)
        (in-syns '())
        (out-syns '())
        (transf (if  in-or-out-neuron? #'(lambda (x) x) *transf*))
        (transf-diff *transf-diff*)
        ) 
    (defun add-in-synaps (s)
      (push s in-syns)
      (incf num-in-syns))
    (defun add-out-synaps (s)
      (push s out-syns)
      (incf num-out-syns))
    (defun upd-out-syns ()
      (if (> num-in-syns num-syns-has-value)
        (incf num-syns-has-value))
      (when (= num-in-syns num-syns-has-value)
        (setf in-val (loop for s in in-syns sum (outval s)))
        (setf out-val (funcall transf in-val))
        (dolist (s out-syns) (setsyn s out-val))
        (setf num-syns-has-value 0)
        ))
    (defun upd-delta ()
      (if (eql in-or-out-neuron? 'in) (return-from upd-delta))
      (if (> num-out-syns num-syns-has-delta-value)
        (incf num-syns-has-delta-value))
      (when (= num-out-syns num-syns-has-delta-value)
        (setf delta
              (* (funcall transf-diff out-val)
                 (loop for s in out-syns sum (synw*d s))))
 
        (dolist (s in-syns) (upd-syn-weight s delta))

        (setf num-syns-has-delta-value 0)
        ))
    
    (defun m(z)
      (cond ((eql z 'add-in-syn) #'add-in-synaps)
            ((eql z 'add-out-syn) #'add-out-synaps)
            ((eql z 'in-syns) in-syns)
            ((eql z 'out-syns) out-syns)
            ((eql z 'upd-out) #'upd-out-syns)
            ((eql z 'upd-delta) #'upd-delta)
            ((eql z 8) delta)
            ((eql z 'in) in-val)
            ((eql z 'out) out-val)
            ((eql z 'name) name)
            ((eql z 'delta) delta)
            ((eql z 'num-syns-has-delta-value) num-syns-has-delta-value)
            ((eql z 'func) transf)
            (t 'error-neurons-args)))
    (push #'m *neurons*)
    #'m))


(defun inval (obj)
  (funcall obj 'in))

(defun outval (obj)
  (funcall obj 'out))

(defun name (obj)
  (funcall obj 'name))

(defun upd-neuron (neuron)
  (funcall (funcall neuron 'upd-out)))
  
(defun add-in-syn (neuron synapse)
  (funcall (funcall neuron 'add-in-syn) synapse))

(defun add-out-syn (neuron synapse)
  (funcall (funcall neuron 'add-out-syn) synapse))

(defun upd-nrn-delta (neuron)
  (funcall (funcall neuron 'upd-delta)))

(defun nrn-delta (neuron)
  (funcall neuron 'delta))

;;Синапс - связной элемент между нейронами, структура по типу нейрона. Замыкания.
(defun synapse (neuron-from neuron-to weight &optional name)
  (let ((in-val 0.0) (out-val 0.0) (expect-val nil) (w*d 0.0) (delta-w 0.0))
    (defun set-weight (x)
      (setf weight x))
    (defun set-value (x)
      (setf in-val x)
      (setf out-val (* x weight))
      (if neuron-to
        (upd-neuron neuron-to)))
    (defun set-expect-value (x)
      (setf expect-val x)
      (setf w*d (- expect-val out-val))
      (upd-nrn-delta neuron-from))
    (defun upd-weight (delta)
      (when neuron-from
        (setf w*d (* weight delta))
        (setf delta-w
              (+
               (* *eps* in-val delta)
               (* *alpha* delta-w)))
        (incf weight delta-w)
        ;(setf w*d (* weight delta))
        (upd-nrn-delta neuron-from)
        ))
        
    (defun m (z)
      (cond
        ((eql z 'neuron-from) neuron-from)
        ((eql z 'neuron-to) neuron-to)
        ((eql z 'weight) weight)
        ((eql z 'set-weight) #'set-weight)
        ((eql z 'expect-val) expect-val)
        ((eql z 'w*d) w*d)
        ((eql z 'in) in-val)
        ((eql z 'out) out-val)
        ((eql z 'name) name)
        ((eql z 'set) #'set-value)
        ((eql z 'set-expect) #'set-expect-value)
        ((eql z 'upd-weight) #'upd-weight)
        ((eql z 'delta-w) delta-w)
        (t 'error)))
    (if neuron-from
      (add-out-syn neuron-from #'m))
    (if neuron-to
      (add-in-syn neuron-to #'m))
    (push #'m *syns*)
    #'m))


(defun synw*d (synapse)
  (funcall synapse 'w*d))

(defun synweight (synapse)
  (funcall synapse 'weight))

(defun setweight (synapse weight)
  (funcall (funcall synapse 'set-weight) weight))

(defun setsyn (synapse val)
  (funcall (funcall synapse 'set) val))

(defun upd-syn-weight (synapse delta)
  (funcall (funcall synapse 'upd-weight) delta))

(defun nn-data (input expect trans-func)
  (let ((input-norm '())
        (expect-norm '())
        (min-in 0.0)
        (max-in 0.0)
        (d-in 0.0)
        (min-expect 0.0)
        (max-expect 0.0)
        (d-expect 0.0)
        (transf nil)
        (transf-diff nil)
        )

    (defun sigmoid-f (x)
      (/ 1 (+ 1 (exp (- x)))))

    (defun sigmoid-diff (out-val)
      (* out-val (- 1 out-val)))

    (defun tanh-f (x)
      (cond
        ((> x 8.5) 1.0)
        ((< x -8.5) -1.0)
        (t
         (/ (- (exp (* 2.0 x)) 1.0)
            (+ (exp (* 2.0 x)) 1.0)))))

    (defun tanh-diff (out)
      (- 1 (* out out)))
        
    (defun min-val (lst)
      (apply #'min (reduce #'append lst)))

    (defun max-val (lst)
      (apply #'max (reduce #'append lst)))
    
    (defun norm (lst xmin dx)
      (loop for lst-el in lst
            collect
            (loop for x in lst-el
                  collect
                  (norm-x x xmin dx))))
 
    (defun norm-x (x xmin dx)
     (- (* 2.0 (/ (- x xmin) dx)) 1))
          
    (defun denorm-x (x xmin dx)
      (+ xmin (/ (* (+ x 1.0) dx) 2.0)))

    (defun norm-i (val) (norm-x val min-in d-in))
    (defun norm-e (val) (norm-x val min-expect d-expect))
    (defun denorm-i (val) (denorm-x val min-in d-in))
    (defun denorm-e (val) (denorm-x val min-expect d-expect))
    
    (defun m (z)
      (cond
        ((eql z 'input) input)
        ((eql z 'expect) expect)
        ((eql z 'input-norm) input-norm)
        ((eql z 'expect-norm) expect-norm)
        ((eql z 'norm-i) #'norm-i)
        ((eql z 'norm-e) #'norm-e)
        ((eql z 'denorm-i) #'denorm-i)
        ((eql z 'denorm-e) #'denorm-e)))

    (cond
      ((eql trans-func 'sigmoid)
       (setf transf #'sigmoid-f)
       (setf transf-diff #'sigmoid-diff))
      ((eql trans-func 'tanh)
       (setf transf #'tanh-f)
       (setf transf-diff #'tanh-diff)))

    (setf *transf* transf)
    (setf *transf-diff* transf-diff)
    
    (setf min-in (min-val input))
    (setf max-in (max-val input))
    (setf d-in (- max-in min-in))
    (setf input-norm (norm input min-in d-in))

    (setf min-expect (min-val expect))
    (setf max-expect (max-val expect))
    (setf d-expect (- max-expect min-expect))
    (setf expect-norm (norm expect min-expect d-expect)))

  #'m
  )

(defun input ()
  (prc *gdata* 'input))

(defun input-norm ()
  (prc *gdata* 'input-norm))

(defun expect ()
  (prc *gdata* 'expect))

(defun expect-norm ()
  (prc *gdata* 'expect-norm))





(defun prin-nrns ()
  (dolist (x *neurons*)
    (format t "name: ~3a   in: ~7,2f    out: ~7,2f   delta: ~5,2f~%"
            (name x) (inval x) (outval x) (prc x 'delta))))

(defun prin-syns ()
  (dolist (x *syns*)
    (format t "syn: ~8a  in: ~7,2f  out: ~7,2f  weight: ~7,2f  w*d: ~7,2f  delta-w: ~5,2f~
~%" (name x) (inval x) (outval x) (synweight x) (prc x 'w*d) (prc x 'delta-w))))

(defun nn-set-input (in-data)
  (loop for x in *in-syns*
        for y in in-data do 
          (prc x 'set y))
  (if *bias-in-syns* (loop for z in *bias-in-syns* do (prc z 'set 1))))


(defun nn-set-expect (expect-data)
  (loop for x in *out-syns*
        for y in expect-data do 
       (prc x 'set-expect y)))

(defun outs () (prcl *out-syns* 'out))

(defun learn-nn (n input expect)
  (loop repeat n do
    (loop for x in input
          for y in expect do
               (progn
                 (nn-set-input x)
                 (nn-set-expect y)
                 ))))



(defvar *rand-weight-from* -1.0)
(defvar *rand-weight-to* 1.0)
(defvar *weights* '())
(defvar *rand-delta* 0.1)

(defvar *gdata* nil)


(defun init (eps alpha)
  (setf *rand-weight-from* -0.5)
  (setf *rand-weight-to* 0.5)
  (setf *rand-delta* 0.3)
  (setf *eps* eps)
  (setf *alpha* alpha)

  ;(setf *input* '((0 0) (1 0) (0 1) (1 1)))
  ;(setf *expect* '((0) (1) (1) (0)))

  ;(setf *input*  (loop for x from 0.0 to (* pi 2) by 0.2 collect (list x)))
  ;(setf *expect*  (loop for x from 0.0 to (* pi 2) by 0.2 collect (list (sin x))))

  (setf *gdata* (nn-data 
                 (loop for x from 0.0 to (* pi 2) by 0.2 collect (list x))
                 (loop for x from 0.0 to (* pi 2) by 0.2 collect (list (sin x)))
                 'tanh))
  
  ;(cr-3l-nn 1 6 1)
  ;(cr-3l-wbias-nn 1 3 1)
  (cr-4l-wbias-nn 1 3 3 1)

  (setf *syns* (nreverse *syns*))
  (setf *neurons* (nreverse *neurons*))
  
  (setf *weights* (gen-weights -0.5 0.5))

  (loop for x in *syns* for w in *weights* do (setweight x w))
  )

(defun gen-weights (rand-from rand-to)
   (loop for x in *syns*
              collect
              (if (not (and (prc x 'neuron-from) (prc x 'neuron-to)))
                1.0
                (random-d rand-from rand-to))))

  
(defun start-nn (eps alpha nepoch)
  (init eps alpha)
  (learn-nn nepoch (prc *gdata* 'input-norm) (prc *gdata* 'expect-norm))
  (prin-in-expect-out (prc *gdata* 'input) (prc *gdata* 'expect))
  )



(defun find-rand< (delta old-val x0 x1)
  (let* ((rn (random-d x0 x1)) (d (abs (- old-val rn))) (rnmax rn) (dmax d))
    (loop repeat 100 do
      (progn
        (if (> d delta)
          (return rn)
          (if (> d dmax)
            (setf dmax d rnmax rn)))
        (setf rn (random-d x0 x1))
        (setf d (abs (- old-val rn))))
          finally (return rnmax))))
                                              



 
(defun learn-wnew-weights (n rand-delta)
  ;(setf *rand-delta* rand-delta)

  (setf *weights*
        (loop for x in *syns*
              collect
              (if (not (and (prc x 'neuron-from) (prc x 'neuron-to)))
                1.0
                (find-rand< rand-delta (prc x 'weight) *rand-weight-from* *rand-weight-to*))))
  
  (loop for x in *syns* for y in *weights* do (prc x 'set-weight y))
  
  (learn-nn n (prc *gdata* 'input-norm) (prc *gdata* 'expect-norm))
  (prin-in-expect-out (prc *gdata* 'input) (prc *gdata* 'expect)))


(defun prin-in-expect-out (in exp)
  (let ((res '()))
  (loop for x in in
        for y in exp do
          (format t "in: ~13a expect: ~15a calc: ~15a delta %: ~15a~%" x y
                  (progn
                    (nn-set-input (mapcar  (prc *gdata* 'norm-i) x))
                    (setf res (mapcar (prc *gdata* 'denorm-e) (prcl *out-syns* 'out))))
                  (mapcar #'(lambda (x1 y1) (* 100 (/ (- x1 y1) (if (< x1 0.0001) 1 x1)))) y res)))))
                  

(defun in-exp-out (x)
 (let ((res '()))
  (format t "in: ~7a expect: ~7a calc: ~7a delta %: ~7a~%" x (mapcar #'sin x)
          (progn
            (nn-set-input (mapcar #'norm-i x))
            (setf res (mapcar (prc *gdata* 'denorm-e) (prcl *out-syns* 'out))))
          (mapcar #'(lambda (x1 y1) (* 100 (/ (- x1 y1) (if (< x1 0.0001) 1 x1)))) (mapcar #'sin x) res))))


;;Создание 3 слойной НС входными, скрытыми, выходными слоями
(defun cr-3l-nn (n-in n-hid n-out)
  (setf *neurons* '())
  (setf *syns* '())
  
  (setf *in-nrns*  (cr-nrns-l n-in "I" 'in))
  (setf *hid-nrns* (cr-nrns-l n-hid "H"))
  (setf *out-nrns* (cr-nrns-l n-out "R"))

  (setf *in-syns* (cr-syns-l nil *in-nrns*)) 
  (cr-syns-l *in-nrns* *hid-nrns*) 
  (cr-syns-l *hid-nrns* *out-nrns*)
  (setf *out-syns* (cr-syns-l *out-nrns* nil))
  )

(defvar hid-1 '())
(defvar hid-2 '())
(defvar bias-1 '())
(defvar bias-2 '())
(defvar bias-3 '())

(defun cr-4l-wbias-nn (num-in num-hid1 num-hid2 num-out)
  (setf *neurons* '())
  (setf *syns* '())
  
  (setf *in-nrns* (cr-nrns-l num-in "I" 'in))
  (setf hid-1 (cr-nrns-l num-hid1 "H0"))
  (setf hid-2 (cr-nrns-l num-hid2 "H1"))
  (setf *out-nrns* (cr-nrns-l num-out "R"))
  (setf bias-1 (cr-nrns-l 1 "B0" 'in))
  (setf bias-2 (cr-nrns-l 1 "B1" 'in))
  (setf bias-3 (cr-nrns-l 1 "B2" 'in))

  (setf *in-syns* (cr-syns-l nil *in-nrns*)) 

  (cr-syns-l *in-nrns* hid-1)
  (cr-syns-l hid-1 hid-2)
  (cr-syns-l hid-2 *out-nrns*)
  
  (setf *bias-in-syns* (nconc  
                        (cr-syns-l nil bias-1)
                        (cr-syns-l nil bias-2)
                        (cr-syns-l nil bias-3)))
  
  (cr-syns-l bias-1 hid-1)
  (cr-syns-l bias-2 hid-2)
  (cr-syns-l bias-3 *out-nrns*)
  
  (setf *out-syns* (cr-syns-l *out-nrns* nil))
 
  )



(defun cr-3l-wbias-nn (n-in-nrns n-hid-nrn n-out-nrn)
  (let ((bias-hid '()) (bias-out '()))  
   (cr-3l-nn n-in-nrns n-hid-nrn n-out-nrn)
   (setf bias-hid (cr-nrns-l 1 "BH" 'in))
   (setf bias-out (cr-nrns-l 1 "BR" 'in))

   (setf *bias-in-syns* (nconc
                         (cr-syns-l nil bias-hid)
                         (cr-syns-l nil bias-out)))
   
   (cr-syns-l bias-hid *hid-nrns*)
   (cr-syns-l bias-out *out-nrns*)

   ;(setf *syns* (nreverse *syns*))
   ;(setf *neurons* (nreverse *neurons*))

   ))
     
(defun cr-nrns-l (num name &optional inlayer?)
  (let ((layer '()))
    (dotimes (i num)
      (push
       (neuron inlayer? (concatenate 'string name (write-to-string i)))
       layer))
    (reverse layer)))

(defun cr-syns-l (nrn-layer-from nrn-layer-to)
  (let ((syns '()))
    (cond ((null nrn-layer-from) ;in-syns
           (dolist (nrn nrn-layer-to)
             (push 
              (synapse nil nrn 1.0
                       (concatenate 'string "->" (prc nrn 'name)))
              syns)))
           ((null nrn-layer-to) ;out-syns
            (dolist (nrn nrn-layer-from)
              (push
               (synapse nrn nil 1.0
                        (concatenate 'string (prc nrn 'name) "->" ))
               syns)))
          (t
           (dolist (nrn-fr nrn-layer-from)
             (dolist (nrn-to nrn-layer-to)
               (push
                (synapse nrn-fr nrn-to 0.0
                         (concatenate 'string
                                      (prc nrn-fr 'name) "->" (prc nrn-to 'name)))
                syns)))))
    (reverse syns)))
    
(defun random-d (from to)
    (+ (random-0 (- to from)) from))

(defun random-0 (x)
  (if (floatp x) (* (/ (random 10000001) 10000000.0) x) (random (+ x 1))))
  
(defun names ()
  (prcl *neurons* 'name))
          
  
