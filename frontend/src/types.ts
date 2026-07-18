import type { WDNData } from './cityData'
import type { SimResult } from './api'

export interface SharedSimState {
  city: string
  wdn: WDNData
  simResult: SimResult | null
  harpResult: SimResult | null
  dinicResult: SimResult | null
  scenario: string
  failedPipes: string[]
  runAt: number
}
