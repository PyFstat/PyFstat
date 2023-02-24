__global__ void cudaTransientFstatRectWindow ( float *input,
                                               unsigned int numAtoms,
                                               unsigned int TAtom,
                                               unsigned int t0_data,
                                               unsigned int win_t0,
                                               unsigned int win_dt0,
                                               unsigned int win_tau,
                                               unsigned int win_dtau,
                                               unsigned int N_tauRange,
                                               float *Fmn
                                             )
{

  /* match CUDA thread indexing and high-level (t0,tau) indexing */
  // assume 1D block, grid setup
  unsigned int m         = blockDim.x * blockIdx.x + threadIdx.x; // t0:  row

  unsigned short input_cols = 7; // must match input matrix!

  /* compute Fstat-atom index i_t0 in [0, numAtoms) */
  unsigned int TAtomHalf = TAtom/2; // integer division
  unsigned int t0 = win_t0 + m * win_dt0;
  /* integer round: floor(x+0.5) */
  int i_tmp = ( t0 - t0_data + TAtomHalf ) / TAtom;
  if ( i_tmp < 0 ) {
    i_tmp = 0;
  }
  unsigned int i_t0 = (unsigned int)i_tmp;
  if ( i_t0 >= numAtoms ) {
    i_t0 = numAtoms - 1;
  }

  float Ad    = 0.0f;
  float Bd    = 0.0f;
  float Cd    = 0.0f;
  float Fa_re = 0.0f;
  float Fa_im = 0.0f;
  float Fb_re = 0.0f;
  float Fb_im = 0.0f;
  unsigned int i_t1_last = i_t0;

  /* INNER loop over timescale-parameter tau
   * NOT parallelized so that we can still use the i_t1_last trick
   * (empirically seems to be faster than 2D CUDA version)
   */
  for ( unsigned int n = 0; n < N_tauRange; n ++ ) {

    if ( (m < N_tauRange) && (n < N_tauRange) ) {

      /* translate n into an atoms end-index
       * for this search interval [t0, t0+Tcoh],
       * giving the index range of atoms to sum over
       */
      unsigned int tau = win_tau + n * win_dtau;

      /* get end-time t1 of this transient-window search */
      unsigned int t1 = t0 + tau;

      /* compute window end-time Fstat-atom index i_t1 in [0, numAtoms)
       * using integer round: floor(x+0.5)
       */
      i_tmp = ( t1 - t0_data + TAtomHalf ) / TAtom  - 1;
      if ( i_tmp < 0 ) {
        i_tmp = 0;
      }
      unsigned int i_t1 = (unsigned int)i_tmp;
      if ( i_t1 >= numAtoms ) {
        i_t1 = numAtoms - 1;
      }

      /* now we have two valid atoms-indices [i_t0, i_t1]
       * spanning our Fstat-window to sum over
       */

      for ( unsigned int i = i_t1_last; i <= i_t1; i ++ ) {
        /* sum up atoms,
         * special optimiziation in the rectangular-window case:
         * just add on to previous tau values,
         * ie re-use the sum over [i_t0, i_t1_last]
         from the pevious tau-loop iteration
         */
        Ad    += input[i*input_cols+0]; // a2_alpha
        Bd    += input[i*input_cols+1]; // b2_alpha
        Cd    += input[i*input_cols+2]; // ab_alpha
        Fa_re += input[i*input_cols+3]; // Fa_alpha_re
        Fa_im += input[i*input_cols+4]; // Fa_alpha_im
        Fb_re += input[i*input_cols+5]; // Fb_alpha_re
        Fb_im += input[i*input_cols+6]; // Fb_alpha_im
        /* keep track of up to where we summed for the next iteration */
        i_t1_last = i_t1 + 1;
      }

      /* get inverse antenna pattern determinant,
       * following safety checks from
       * XLALComputeAntennaPatternSqrtDeterminant()
       * and estimateAntennaPatternConditionNumber()
       */
      float sumAB  = Ad + Bd;
      float diffAB = Ad - Bd;
      float disc   = sqrt ( diffAB*diffAB + 4.0 * Cd*Cd );
      float denom = sumAB - disc;
      float cond = (denom > 0) ? ((sumAB + disc) / denom) : INFINITY;
      float DdInv = 0.0f;
      if ( cond < 1e4 ) {
        DdInv = 1.0 / ( Ad * Bd - Cd * Cd );
      }

      /* matching compute_fstat_from_fa_fb
       * including default fallback = 0.5*E[2F] in noise
       * when DdInv == 0 due to ill-conditionness of M_munu
       */
      float F = 2;
      if ( DdInv > 0 ) {
        F  = DdInv * (   Bd * ( Fa_re*Fa_re + Fa_im*Fa_im )
                       + Ad * ( Fb_re*Fb_re + Fb_im*Fb_im )
                       - 2.0 * Cd * ( Fa_re * Fb_re + Fa_im * Fb_im )
                     );
      }

      /* store result in Fstat-matrix
       * at unraveled index of element {m,n}
       */
      unsigned int outidx = m * N_tauRange + n;
      Fmn[outidx] = F;

    } // if ( (m < N_tauRange) && (n < N_tauRange) )

  } // for ( unsigned int n = 0; n < N_tauRange; n ++ )

} // cudaTransientFstatRectWindow()
